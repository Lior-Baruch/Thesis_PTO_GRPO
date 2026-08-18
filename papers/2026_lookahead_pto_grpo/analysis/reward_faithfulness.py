"""reward_faithfulness.py — is the partial-conversation TRAINING reward a faithful proxy for the
full-conversation EVAL, and does K-turn look-ahead make it more faithful?

The tracked EDA renders this only as a curve (``5_training/reward_reliability_curve.png``); here it
becomes TABLES with bootstrap CIs, a stated unit, a matched-policy cut, a persona-cooperation cut, and
the proxy-vs-eval level table.

UNIT (restated in every caption)
  * A *branch row* = one branch point of the training run: a conversation-so-far ("prefix", n_turns
    utterances therapist+patient combined, ending on a patient turn) plus M=8 (PTO) / G=8 (GRPO)
    therapist completions sampled from the iter-start policy pi_{train_iter-1}. Each completion was
    scored AT TRAINING TIME by the training oracle (gpt-4o-mini, mean of Q1+Q2) on
    ``prefix + completion`` (K=0) or ``prefix + completion + 5 simulated turns`` (K=5).
    ``proxy_score`` = the score of the CHOSEN candidate = the arg-max candidate (PTO: the one appended
    to the greedy trunk; GRPO: the recorder marks the arg-max). Read from ``generations.jsonl`` —
    NO new oracle calls (``eda_analysis.training.load_branch_reliability(which="chosen")``).
  * ``n_turns`` = utterances in the prefix BEFORE the scored completion (MCL=12 is the shortest cut).
    The text the oracle actually saw has n_turns+1 (+K) utterances.
  * ``eval_score`` = full-conversation Q1Q2 of the eval conversation the prefix was cut from
    (``model_iter_{train_iter-1}``, joined on ``conversation_id == file_index``; the shuffle seed is
    shared, so this is also the same persona). For GRPO the prefix IS a slice of that eval
    conversation at every n_turns; for PTO (greedy trunk) it shares the first MCL=12 utterances and
    afterwards follows the greedy best-of-M continuation, so the two diverge as n_turns grows.
  * ``agreement`` (per arm, per n_turns) = the fraction of conversation PAIRS, formed within one
    (arm, eval_iter, n_turns) cell, whose proxy-score ordering matches their eval-score ordering
    (ties dropped), pooled (summed counts) over eval_iters. 0.5 = chance. Point estimates reproduce
    ``eda_analysis.stats.rank_agreement_by_nturns`` exactly (asserted).
  * CIs = 95% percentile CLUSTER bootstrap: conversations are resampled with replacement within each
    (arm, eval_iter) model state (the same resample applied to every n_turns bin of that state, so
    pooled-over-bins numbers are cluster-correct too); B=1000, seed 0.
  * K-contrast sign: ``+ => K=0 higher`` (delta = K0 - K5), mirroring the paper's convention. NOTE that
    for faithfulness a NEGATIVE delta means look-ahead helped.
  * Graders: the proxy is ALWAYS the training oracle's score (it cannot be re-graded). The eval side is
    computed under BOTH graders: primary = gpt-4o-mini (the same oracle -> same-grader faithfulness),
    held-out = Claude Haiku 4.5 (cross-grader faithfulness). Never averaged.
  * GRPO_LA5 is right-censored at iteration 5 (train_iter 1..5).

Outputs (all prefixed ``reward_faithfulness_``):
  tables/   curve.md (+ curve_heldout.md, curve_long.csv), curve_by_iter.md,
            matched_policy.md (+ matched_policy_tests.md), by_coop.md, levels.md (+ levels_rho.md)
  figures/  fig.png (primary grader), fig_heldout.png
  analysis/out/reward_faithfulness.json

Dev note: setting env ``RF_DEV_CACHE=<dir>`` holding ``BR.parquet`` + ``SC_primary.parquet`` +
``SC_heldout.parquet`` short-circuits the ~3 min loaders; unset (the default) uses the real loaders.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats as sps  # noqa: E402

from eda_analysis import training, stats as est, data as edata  # noqa: E402

SCRIPT = "reward_faithfulness"
N_BOOT = 1000
SEED = 0
METRIC = "Q1Q2"
METHODS = ["PTO", "GRPO"]
JUDGE_KEYS = ["primary", "heldout"]
JS = {"primary": C.JUDGE_SHORT[C.PRIMARY], "heldout": C.JUDGE_SHORT[C.HELDOUT]}
COARSE = [("12-20", 12, 20), ("22-34", 22, 34), ("36-50", 36, 50)]
COOP_LABEL = {"High": "Cooperative", "StartLowAndChangesToHigh": "Warms up", "Low": "Resistant"}
COOP_ORDER = ["Cooperative", "Warms up", "Resistant"]

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
CENSOR_NOTE = "GRPO_LA5 is right-censored at iteration 5 (train_iter 1..5, eval_iter 0..4)."
GRADER_NOTE = ("Proxy = the training oracle (gpt-4o-mini) by construction; eval side under the grader named in "
               "the table (primary = gpt-4o-mini, held-out = Claude Haiku 4.5). Never averaged across graders.")


def fmt_ci(a, lo, hi, nd=3):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return ""
    return f"{a:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# ── data ─────────────────────────────────────────────────────────────────────

def eval_frame(scores_long: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """(arm, eval_iter, conversation_id) -> eval_score (+ persona_id, coop) from a scores_long frame.
    Mirrors the join in ``stats.rank_agreement_by_nturns`` (iteration -> eval_iter, file_index -> conversation_id)."""
    d = scores_long[scores_long["questionnaire"] == metric]
    d = d.rename(columns={"iteration": "eval_iter", "file_index": "conversation_id"})
    ev = (d.groupby(["arm", "eval_iter", "conversation_id"], as_index=False)
           .agg(eval_score=("score", "mean"), persona_id=("persona_id", "first"),
                coop_raw=("cooperation_level", "first")))
    ev["coop"] = ev["coop_raw"].map(COOP_LABEL)
    return ev


class AgreementBoot:
    """Pairwise sign-agreement between proxy and eval with a conversation-level cluster bootstrap.

    ``df`` columns: arm, eval_iter, conversation_id, n_turns, proxy_score, eval_score.
    Cells are keyed (arm, eval_iter, n_turns); each holds the point counts (conc, tot) over pairs and
    the B bootstrap replicate counts. Any aggregate = sum of counts over a set of cells, so per-bin,
    per-iteration and overall numbers all come from the same replicates.
    """

    def __init__(self, df: pd.DataFrame, *, B: int = N_BOOT, seed: int = SEED):
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


def delta_ci(bootsA, bootsB, a, b):
    """delta = a - b with a percentile CI from independent replicate arrays."""
    d = bootsA - bootsB
    if not np.isfinite(d).any():
        return dict(delta=np.nan, d_lo=np.nan, d_hi=np.nan)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return dict(delta=a - b, d_lo=float(lo), d_hi=float(hi))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    L = C.Ledger(SCRIPT)
    C.style()
    pal = C.palette()

    arms = edata.discover_arms()
    arms = [a for a in arms if a.label in C.ARMS]
    dev = os.environ.get("RF_DEV_CACHE")            # optional dev-only parquet cache; unset = real loaders
    if dev and os.path.exists(os.path.join(dev, "BR.parquet")):
        BR = pd.read_parquet(os.path.join(dev, "BR.parquet"))
        SC = {j: pd.read_parquet(os.path.join(dev, f"SC_{j}.parquet")) for j in JUDGE_KEYS}
    else:
        BR = training.load_branch_reliability(arms, which="chosen")
        SC = C.load_scores_both()
    EV = {j: eval_frame(SC[j]) for j in JUDGE_KEYS}
    DF = {j: BR.merge(EV[j][["arm", "eval_iter", "conversation_id", "eval_score", "persona_id", "coop"]],
                      on=["arm", "eval_iter", "conversation_id"], how="inner") for j in JUDGE_KEYS}
    for j in JUDGE_KEYS:
        print(f"[{j}] branch rows {len(BR)} -> joined {len(DF[j])} "
              f"({DF[j]['conversation_id'].nunique()} convs, arms {sorted(DF[j]['arm'].unique())})")

    # ── self-check: point estimates reproduce the tracked EDA function exactly ─────────────────
    AB = {j: AgreementBoot(DF[j][["arm", "eval_iter", "conversation_id", "n_turns", "proxy_score", "eval_score"]])
          for j in JUDGE_KEYS}
    ref = est.rank_agreement_by_nturns(BR, SC["primary"], metric=METRIC, min_pairs=20)
    max_dev = 0.0
    for _, r in ref.iterrows():
        mine = AB["primary"].agg(AB["primary"].keys(arm=r["arm"], nt_lo=int(r["n_turns"]), nt_hi=int(r["n_turns"])))
        max_dev = max(max_dev, abs(mine["agreement"] - r["agreement"]))
        assert mine["n_pairs"] == r["n_pairs"], (r["arm"], r["n_turns"], mine["n_pairs"], r["n_pairs"])
    assert max_dev < 1e-9, max_dev
    print(f"[selfcheck] point estimates match stats.rank_agreement_by_nturns on {len(ref)} (arm,n_turns) bins "
          f"(max |dev| = {max_dev:.2e})")
    L.put("selfcheck.max_abs_dev_vs_eda_function", max_dev,
          source="stats.rank_agreement_by_nturns(BR, primary scores) vs this script's point estimates")

    all_bins = sorted(BR["n_turns"].unique())
    train_iters = {a: sorted(BR.loc[BR["arm"] == a, "train_iter"].unique()) for a in C.ARMS}

    # ── 1) curve: pooled over iterations, per n_turns bin, with CIs ──────────────────────────────
    # series = (arm, iteration subset). GRPO_LA0 appears twice: its full support (1-10) and the
    # 1-5 subset matched to GRPO_LA5's censored support, so the K read is like-for-like.
    SERIES = [("PTO_LA0", "1-10", None), ("PTO_LA5", "1-10", None), ("GRPO_LA0", "1-10", None),
              ("GRPO_LA0", "1-5", {0, 1, 2, 3, 4}), ("GRPO_LA5", "1-5", None)]
    long_rows = []
    for j in JUDGE_KEYS:
        ab = AB[j]
        for arm, iters_lab, eis in SERIES:
            for nt in all_bins + ["all"] + [c[0] for c in COARSE]:
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
                if r["n_pairs"] < 20:
                    continue
                long_rows.append(dict(judge=JS[j], arm=arm, iters=iters_lab, method=C.method_of(arm), K=C.k_of(arm),
                                      n_turns=str(nt), agreement=r["agreement"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                                      n_pairs=r["n_pairs"], n_iters=len({k[1] for k in ks}),
                                      n_convs_mean=r["n_convs_mean"]))
    CURVE = pd.DataFrame(long_rows)
    CURVE_FULL = CURVE[~((CURVE["arm"] == "GRPO_LA0") & (CURVE["iters"] == "1-5"))]   # one row per arm
    C.save_table(CURVE, f"{SCRIPT}_curve_long",
                 caption=("Long form of the reward-faithfulness curve: sign-agreement between the training proxy "
                          "and the full-conversation eval Q1Q2, per (grader of the eval side, arm, n_turns bin), pooled "
                          "over training iterations; rows n_turns='all' pool every bin, '12-20'/'22-34'/'36-50' pool "
                          f"coarse ranges. {UNIT_NOTE} {GRADER_NOTE} {CENSOR_NOTE}"))

    def wide_curve(judge_short):
        d = CURVE[CURVE["judge"] == judge_short]
        rows = []
        for nt in [str(b) for b in all_bins] + ["12-20", "22-34", "36-50", "all"]:
            row = {"n_turns": nt}
            for arm, iters_lab, _ in SERIES:
                col = f"{arm} (iters {iters_lab})"
                s = d[(d["arm"] == arm) & (d["iters"] == iters_lab) & (d["n_turns"] == nt)]
                if s.empty:
                    row[col] = ""
                    row[f"{col} pairs"] = ""
                else:
                    s = s.iloc[0]
                    row[col] = fmt_ci(s["agreement"], s["ci_lo"], s["ci_hi"])
                    row[f"{col} pairs"] = int(s["n_pairs"])
            rows.append(row)
        return pd.DataFrame(rows)

    C.save_table(wide_curve(JS["primary"]), f"{SCRIPT}_curve",
                 caption=("Reward faithfulness vs partial-conversation length — same-grader (eval side = the training "
                          "oracle, gpt-4o-mini). Cells: agreement [95% CI]; *_pairs = number of conversation pairs pooled "
                          f"over iterations (the column header names the train_iter range pooled). {UNIT_NOTE} "
                          f"{CENSOR_NOTE} GRPO_LA0 is shown both on its full support (iters 1-10) and restricted to iters "
                          "1-5 (like-for-like with GRPO_LA5). Bins with n_pairs < 20 are not reported."))
    C.save_table(wide_curve(JS["heldout"]), f"{SCRIPT}_curve_heldout",
                 caption=("Reward faithfulness vs partial-conversation length — CROSS-grader (eval side = the held-out "
                          "judge, Claude Haiku 4.5; the proxy is still the training oracle's score). Cells: agreement "
                          f"[95% CI]; *pairs = conversation pairs pooled over the train_iter range in the column header. "
                          f"{UNIT_NOTE} {CENSOR_NOTE} GRPO_LA0 is shown both on its full support (iters 1-10) and restricted to iters 1-5."))

    # ledger: headline bins
    for j in JUDGE_KEYS:
        for arm, iters_lab, _ in SERIES:
            for nt in ["12", "20", "30", "40", "12-20", "22-34", "36-50", "all"]:
                s = CURVE[(CURVE["judge"] == JS[j]) & (CURVE["arm"] == arm) & (CURVE["iters"] == iters_lab) & (CURVE["n_turns"] == nt)]
                if s.empty:
                    continue
                s = s.iloc[0]
                L.put(f"curve.{j}.{arm}.iters{iters_lab}.nt{nt}",
                      dict(agreement=round(float(s["agreement"]), 3), ci_lo=round(float(s["ci_lo"]), 3),
                           ci_hi=round(float(s["ci_hi"]), 3), n_pairs=int(s["n_pairs"])),
                      source=f"tables/{SCRIPT}_curve_long.md row judge={JS[j]}, arm={arm}, iters={iters_lab}, n_turns={nt}")

    # ── K contrast per bin under three iteration CUTS (computed after the per-iter block) ────────
    def cut_iters(m: str, cut: str) -> set:
        """eval_iters used by a cut: train_iter_1 -> {0}; iters_1-5 -> {0..4}; matched_iters -> every
        train_iter present in BOTH K arms of the method (PTO 1..10, GRPO 1..5)."""
        both = set(train_iters[f"{m}_LA0"]) & set(train_iters[f"{m}_LA5"])
        if cut == "train_iter_1":
            ti = {1}
        elif cut == "iters_1-5":
            ti = {t for t in both if t <= 5}
        else:
            ti = both
        return {int(t) - 1 for t in ti}
    CUTS = ["train_iter_1", "iters_1-5", "matched_iters"]

    # ── 1b) per train_iter ────────────────────────────────────────────────────────────────────────
    by_iter_rows = []
    for arm in C.ARMS:
        for ti in train_iters[arm]:
            ei = int(ti) - 1
            row = dict(arm=arm, method=C.method_of(arm), K=C.k_of(arm), train_iter=int(ti), eval_iter=ei)
            for j in JUDGE_KEYS:
                ab = AB[j]
                r = ab.agg(ab.keys(arm=arm, eval_iters={ei}))
                tag = "" if j == "primary" else "_heldout"
                row[f"agreement{tag}"] = r["agreement"]; row[f"ci_lo{tag}"] = r["ci_lo"]; row[f"ci_hi{tag}"] = r["ci_hi"]
                row[f"n_pairs{tag}"] = r["n_pairs"]
                if j == "primary":
                    row["n_convs"] = int(DF[j][(DF[j]["arm"] == arm) & (DF[j]["eval_iter"] == ei)]["conversation_id"].nunique())
                    row["n_bins"] = r["n_cells"]
                    for lab, lo, hi in COARSE:
                        rc = ab.agg(ab.keys(arm=arm, eval_iters={ei}, nt_lo=lo, nt_hi=hi))
                        row[f"agr_{lab}"] = rc["agreement"]; row[f"lo_{lab}"] = rc["ci_lo"]; row[f"hi_{lab}"] = rc["ci_hi"]
                        row[f"pairs_{lab}"] = rc["n_pairs"]
            by_iter_rows.append(row)
    BYITER = pd.DataFrame(by_iter_rows)
    # K0-K5 per (method, train_iter), overall bins, primary + heldout
    dk_rows = []
    for j in JUDGE_KEYS:
        ab = AB[j]
        for m in METHODS:
            for ti in train_iters[f"{m}_LA5"]:
                ei = int(ti) - 1
                r0, r5 = ab.agg(ab.keys(arm=f"{m}_LA0", eval_iters={ei})), ab.agg(ab.keys(arm=f"{m}_LA5", eval_iters={ei}))
                d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
                dk_rows.append(dict(judge=JS[j], method=m, train_iter=int(ti), eval_iter=ei,
                                    agr_K0=r0["agreement"], agr_K5=r5["agreement"], delta_K0_minus_K5=d["delta"],
                                    d_lo=d["d_lo"], d_hi=d["d_hi"], pairs_K0=r0["n_pairs"], pairs_K5=r5["n_pairs"]))
    DK = pd.DataFrame(dk_rows)

    md = BYITER.copy()
    md["agreement [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(md["agreement"], md["ci_lo"], md["ci_hi"])]
    md["agreement_heldout [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(md["agreement_heldout"], md["ci_lo_heldout"], md["ci_hi_heldout"])]
    for lab, _, _ in COARSE:
        md[f"agr {lab} [CI]"] = [fmt_ci(a, l, h) if n >= 20 else "" for a, l, h, n in
                                 zip(md[f"agr_{lab}"], md[f"lo_{lab}"], md[f"hi_{lab}"], md[f"pairs_{lab}"])]
    md_cols = ["arm", "train_iter", "eval_iter", "n_convs", "n_pairs", "agreement [CI]"] + [f"agr {lab} [CI]" for lab, _, _ in COARSE] + ["agreement_heldout [CI]"]
    C.save_table(md[md_cols], f"{SCRIPT}_curve_by_iter",
                 caption=("Reward faithfulness per training iteration (all n_turns bins pooled, plus three coarse "
                          "ranges): agreement [95% CI] with the eval side under the training oracle (gpt-4o-mini) and, "
                          "last column, under the held-out judge (Claude Haiku 4.5); coarse-range cells with < 20 pairs are blank. train_iter n branches from policy "
                          "pi_{n-1}, whose eval conversations are model_iter_{n-1} = eval_iter; train_iter 1 = the BASE "
                          f"policy for every arm (a matched-policy row). {UNIT_NOTE} {CENSOR_NOTE}"))
    BYITER.to_csv(C.TABLES / f"{SCRIPT}_curve_by_iter_long.csv", index=False)
    dkm = DK.copy()
    dkm["delta_K0_minus_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(dkm["delta_K0_minus_K5"], dkm["d_lo"], dkm["d_hi"])]
    C.save_table(dkm[["judge", "method", "train_iter", "eval_iter", "agr_K0", "agr_K5", "delta_K0_minus_K5 [CI]", "pairs_K0", "pairs_K5"]],
                 f"{SCRIPT}_k_by_iter",
                 caption=("K=0 vs K=5 faithfulness per training iteration (all bins pooled), per method and eval-side "
                          f"grader. {SIGN_NOTE} CI = percentile of the difference of independent cluster-bootstrap replicates "
                          "(the two arms are different conversation draws). Only train_iter 1 samples the SAME policy in both K "
                          f"arms; later rows compare diverged policies. {UNIT_NOTE} {CENSOR_NOTE}"))
    for _, r in DK.iterrows():
        L.put(f"k_by_iter.{'primary' if r['judge']==JS['primary'] else 'heldout'}.{r['method']}.train_iter{int(r['train_iter'])}",
              dict(agr_K0=round(float(r["agr_K0"]), 3), agr_K5=round(float(r["agr_K5"]), 3),
                   delta_K0_minus_K5=round(float(r["delta_K0_minus_K5"]), 3), d_lo=round(float(r["d_lo"]), 3), d_hi=round(float(r["d_hi"]), 3)),
              source=f"tables/{SCRIPT}_k_by_iter.md row judge={r['judge']}, method={r['method']}, train_iter={int(r['train_iter'])}")
    for _, r in BYITER.iterrows():
        L.put(f"by_iter.{r['arm']}.train_iter{int(r['train_iter'])}",
              dict(agreement=round(float(r["agreement"]), 3), ci_lo=round(float(r["ci_lo"]), 3), ci_hi=round(float(r["ci_hi"]), 3),
                   n_pairs=int(r["n_pairs"]), n_convs=int(r["n_convs"]),
                   agreement_heldout=round(float(r["agreement_heldout"]), 3)),
              source=f"tables/{SCRIPT}_curve_by_iter.md row arm={r['arm']}, train_iter={int(r['train_iter'])}")

    # ── 2) K contrast in faithfulness: matched-policy cut (train_iter 1) + two pooled cuts ─────────
    mp_rows, test_rows, sum_rows = [], [], []
    for j in JUDGE_KEYS:
        ab = AB[j]
        for m in METHODS:
            a0, a5 = f"{m}_LA0", f"{m}_LA5"
            for cut in CUTS:
                eis = cut_iters(m, cut)
                deltas = []
                for nt in all_bins + [c[0] for c in COARSE] + ["all"]:
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
                    if r0["n_pairs"] < 20 or r5["n_pairs"] < 20:
                        continue
                    d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
                    mp_rows.append(dict(judge=JS[j], method=m, cut=cut, n_turns=str(nt),
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
                test_rows.append(dict(judge=JS[j], method=m, cut=cut, n_bins=len(deltas),
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
                    if q0["n_pairs"] < 20 or q5["n_pairs"] < 20:
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
                    wi = sps.wilcoxon(per_it, zero_method="wilcox")
                    wi_p = float(wi.pvalue)
                else:
                    wi_p = np.nan
                sum_rows.append(dict(judge=JS[j], method=m, cut=cut, n_iters=len(per_it),
                                     train_iters=",".join(str(e + 1) for e in sorted(eis)),
                                     agr_K0=r0["agreement"], K0_lo=r0["ci_lo"], K0_hi=r0["ci_hi"], pairs_K0=r0["n_pairs"],
                                     agr_K5=r5["agreement"], K5_lo=r5["ci_lo"], K5_hi=r5["ci_hi"], pairs_K5=r5["n_pairs"],
                                     delta_pooled=dp["delta"], dp_lo=dp["d_lo"], dp_hi=dp["d_hi"],
                                     delta_iter_mean=float(per_it.mean()) if len(per_it) else np.nan, di_lo=float(it_lo), di_hi=float(it_hi),
                                     iters_K5_more_faithful=int((per_it < 0).sum()), iters_K0_more_faithful=int((per_it > 0).sum()),
                                     wilcoxon_over_iters_p=wi_p))
    MP = pd.DataFrame(mp_rows)
    TESTS = pd.DataFrame(test_rows)
    KSUM = pd.DataFrame(sum_rows)
    CUT_NOTE = ("Cuts: cut=train_iter_1 = MATCHED POLICY — both K arms of a method branch from the SAME base policy pi_0 "
                "(eval side = that arm's independent base draw, model_iter_0), so K=0 vs K=5 is free of policy divergence; "
                "cut=iters_1-5 pools train_iter 1..5 (GRPO_LA5's full support; policies have diverged); cut=matched_iters "
                "pools every train_iter present in BOTH K arms (PTO 1..10, GRPO 1..5).")
    mpm = MP.copy()
    mpm["agr_K0 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(mpm["agr_K0"], mpm["K0_lo"], mpm["K0_hi"])]
    mpm["agr_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(mpm["agr_K5"], mpm["K5_lo"], mpm["K5_hi"])]
    mpm["delta_K0_minus_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(mpm["delta_K0_minus_K5"], mpm["d_lo"], mpm["d_hi"])]
    C.save_table(mpm[["judge", "method", "cut", "n_turns", "agr_K0 [CI]", "pairs_K0", "agr_K5 [CI]", "pairs_K5", "delta_K0_minus_K5 [CI]"]],
                 f"{SCRIPT}_matched_policy",
                 caption=("K=0 vs K=5 reward faithfulness per n_turns bin under three iteration cuts. " + CUT_NOTE + " "
                          f"{SIGN_NOTE} Per-bin CI = percentile of the difference of independent cluster-bootstrap replicates "
                          f"(the two arms are different conversation draws). Bins with < 20 pairs in either arm are dropped. "
                          f"{UNIT_NOTE} {GRADER_NOTE} {CENSOR_NOTE}"))
    MP.to_csv(C.TABLES / f"{SCRIPT}_matched_policy_long.csv", index=False)
    C.save_table(TESTS, f"{SCRIPT}_matched_policy_tests",
                 caption=("Wilcoxon signed-rank test over n_turns BINS (paired by bin; the per-bin deltas of "
                          f"{SCRIPT}_matched_policy) of delta = agreement(K0) - agreement(K5). {SIGN_NOTE} " + CUT_NOTE + " "
                          "Bins are NOT independent observations (the same conversations feed neighbouring bins), so read p as "
                          "descriptive; the per-bin CIs and the summary table carry the inference. n_bins = bins with >= 20 pairs in both arms."))
    ksm = KSUM.copy()
    ksm["agr_K0 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(ksm["agr_K0"], ksm["K0_lo"], ksm["K0_hi"])]
    ksm["agr_K5 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(ksm["agr_K5"], ksm["K5_lo"], ksm["K5_hi"])]
    ksm["delta pooled pairs [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(ksm["delta_pooled"], ksm["dp_lo"], ksm["dp_hi"])]
    ksm["delta iter-mean [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(ksm["delta_iter_mean"], ksm["di_lo"], ksm["di_hi"])]
    C.save_table(ksm[["judge", "method", "cut", "train_iters", "agr_K0 [CI]", "pairs_K0", "agr_K5 [CI]", "pairs_K5",
                      "delta pooled pairs [CI]", "delta iter-mean [CI]", "iters_K5_more_faithful", "iters_K0_more_faithful", "wilcoxon_over_iters_p"]],
                 f"{SCRIPT}_k_summary",
                 caption=("SUMMARY of the K contrast in reward faithfulness (all n_turns bins pooled) per eval-side grader, "
                          "method and iteration cut. " + CUT_NOTE + " 'delta pooled pairs' weights every conversation pair equally "
                          "(arms with more branch points at some iterations weigh those iterations more); 'delta iter-mean' is the "
                          "mean of the per-iteration K0-K5 deltas (equal weight per training iteration; CI from the same replicates), "
                          "with a Wilcoxon over iterations (n_iters >= 5 only) and the count of iterations on each side. "
                          f"{SIGN_NOTE} {UNIT_NOTE} {GRADER_NOTE} {CENSOR_NOTE}"))
    KSUM.to_csv(C.TABLES / f"{SCRIPT}_k_summary_long.csv", index=False)
    for _, r in KSUM.iterrows():
        jk = "primary" if r["judge"] == JS["primary"] else "heldout"
        L.put(f"k_summary.{jk}.{r['method']}.{r['cut']}",
              dict(train_iters=r["train_iters"], agr_K0=round(float(r["agr_K0"]), 3), agr_K5=round(float(r["agr_K5"]), 3),
                   delta_pooled=round(float(r["delta_pooled"]), 3), dp_lo=round(float(r["dp_lo"]), 3), dp_hi=round(float(r["dp_hi"]), 3),
                   delta_iter_mean=round(float(r["delta_iter_mean"]), 3), di_lo=round(float(r["di_lo"]), 3), di_hi=round(float(r["di_hi"]), 3),
                   n_iters=int(r["n_iters"]), iters_K5_more_faithful=int(r["iters_K5_more_faithful"]),
                   wilcoxon_over_iters_p=(None if np.isnan(r["wilcoxon_over_iters_p"]) else round(float(r["wilcoxon_over_iters_p"]), 4)),
                   pairs_K0=int(r["pairs_K0"]), pairs_K5=int(r["pairs_K5"])),
              source=f"tables/{SCRIPT}_k_summary.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}")
    for _, r in TESTS.iterrows():
        jk = "primary" if r["judge"] == JS["primary"] else "heldout"
        L.put(f"matched_policy_test.{jk}.{r['method']}.{r['cut']}",
              dict(n_bins=int(r["n_bins"]), bins_K5_more_faithful=int(r["bins_K5_more_faithful"]),
                   mean_delta=round(float(r["mean_delta"]), 3), median_delta=round(float(r["median_delta"]), 3),
                   wilcoxon_p=round(float(r["wilcoxon_p"]), 4)),
              source=f"tables/{SCRIPT}_matched_policy_tests.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}")
    for _, r in MP.iterrows():
        if r["n_turns"] in ("all", "12", "12-20", "22-34", "36-50"):
            jk = "primary" if r["judge"] == JS["primary"] else "heldout"
            L.put(f"matched_policy.{jk}.{r['method']}.{r['cut']}.nt{r['n_turns']}",
                  dict(agr_K0=round(float(r["agr_K0"]), 3), agr_K5=round(float(r["agr_K5"]), 3),
                       delta_K0_minus_K5=round(float(r["delta_K0_minus_K5"]), 3), d_lo=round(float(r["d_lo"]), 3),
                       d_hi=round(float(r["d_hi"]), 3), pairs_K0=int(r["pairs_K0"]), pairs_K5=int(r["pairs_K5"])),
                  source=f"tables/{SCRIPT}_matched_policy.md row judge={r['judge']}, method={r['method']}, cut={r['cut']}, n_turns={r['n_turns']}")

    # ── 3) by cooperation level (pairs formed WITHIN a cooperation stratum) ───────────────────────
    coop_rows = []
    d0 = DF["primary"]
    for coop in COOP_ORDER:
        sub = d0[d0["coop"] == coop][["arm", "eval_iter", "conversation_id", "n_turns", "proxy_score", "eval_score"]]
        abc = AgreementBoot(sub, seed=SEED + 1 + COOP_ORDER.index(coop))
        for arm in C.ARMS:
            r = abc.agg(abc.keys(arm=arm))
            row = dict(coop=coop, arm=arm, method=C.method_of(arm), K=C.k_of(arm),
                       n_convs_mean=r["n_convs_mean"], agreement=r["agreement"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_pairs=r["n_pairs"])
            for lab, lo, hi in COARSE:
                rc = abc.agg(abc.keys(arm=arm, nt_lo=lo, nt_hi=hi))
                row[f"agr_{lab}"] = rc["agreement"]; row[f"lo_{lab}"] = rc["ci_lo"]; row[f"hi_{lab}"] = rc["ci_hi"]; row[f"pairs_{lab}"] = rc["n_pairs"]
            # matched-policy (train_iter 1) within stratum
            r1 = abc.agg(abc.keys(arm=arm, eval_iters={0}))
            row["agr_iter1"] = r1["agreement"]; row["lo_iter1"] = r1["ci_lo"]; row["hi_iter1"] = r1["ci_hi"]; row["pairs_iter1"] = r1["n_pairs"]
            coop_rows.append(row)
        # K deltas within stratum (all iters + iter1)
        for m in METHODS:
            r0, r5 = abc.agg(abc.keys(arm=f"{m}_LA0")), abc.agg(abc.keys(arm=f"{m}_LA5"))
            d = delta_ci(r0["boots"], r5["boots"], r0["agreement"], r5["agreement"])
            r0i, r5i = abc.agg(abc.keys(arm=f"{m}_LA0", eval_iters={0})), abc.agg(abc.keys(arm=f"{m}_LA5", eval_iters={0}))
            di = delta_ci(r0i["boots"], r5i["boots"], r0i["agreement"], r5i["agreement"])
            L.put(f"by_coop.primary.{coop}.{m}.delta_K0_minus_K5",
                  dict(all_iters=dict(delta=round(d["delta"], 3), lo=round(d["d_lo"], 3), hi=round(d["d_hi"], 3)),
                       train_iter_1=dict(delta=round(di["delta"], 3), lo=round(di["d_lo"], 3), hi=round(di["d_hi"], 3))),
                  source=f"tables/{SCRIPT}_by_coop.md rows coop={coop}, arms {m}_LA0/{m}_LA5 (delta computed from the same replicates)")
    COOP = pd.DataFrame(coop_rows)
    cm = COOP.copy()
    cm["agreement [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm["agreement"], cm["ci_lo"], cm["ci_hi"])]
    for lab, _, _ in COARSE:
        cm[f"agr {lab} [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm[f"agr_{lab}"], cm[f"lo_{lab}"], cm[f"hi_{lab}"])]
    cm["agr train_iter_1 [CI]"] = [fmt_ci(a, l, h) for a, l, h in zip(cm["agr_iter1"], cm["lo_iter1"], cm["hi_iter1"])]
    cm["n_convs_mean"] = cm["n_convs_mean"].round(1)
    C.save_table(cm[["coop", "arm", "n_convs_mean", "n_pairs", "agreement [CI]"] + [f"agr {lab} [CI]" for lab, _, _ in COARSE] + ["agr train_iter_1 [CI]", "pairs_iter1"]],
                 f"{SCRIPT}_by_coop",
                 caption=("Reward faithfulness by patient cooperation level (32 personas each: Cooperative = 'High', "
                          "Warms up = 'StartLowAndChangesToHigh', Resistant = 'Low'; persona attached to each branch "
                          "row via the eval conversation's file_index -> persona_id). Pairs are formed WITHIN a "
                          "cooperation stratum, so the statistic asks whether the training proxy ranks same-cooperation "
                          "conversations like the full-conversation eval does (the easy between-strata ordering is "
                          "removed). Eval side = the training oracle (gpt-4o-mini). agreement [CI] pools all "
                          "n_turns and iterations; the coarse-range columns split by cut length; the last two columns are "
                          f"the matched-policy (train_iter 1, base policy) cut. n_convs_mean = mean conversations per "
                          f"(eval_iter, n_turns) cell. {UNIT_NOTE} {CENSOR_NOTE}"))
    COOP.to_csv(C.TABLES / f"{SCRIPT}_by_coop_long.csv", index=False)
    for _, r in COOP.iterrows():
        L.put(f"by_coop.primary.{r['coop']}.{r['arm']}",
              dict(agreement=round(float(r["agreement"]), 3), ci_lo=round(float(r["ci_lo"]), 3), ci_hi=round(float(r["ci_hi"]), 3),
                   n_pairs=int(r["n_pairs"]), agr_train_iter_1=round(float(r["agr_iter1"]), 3)),
              source=f"tables/{SCRIPT}_by_coop.md row coop={r['coop']}, arm={r['arm']}")

    # ── 4) proxy-vs-eval LEVELS per arm x iteration ──────────────────────────────────────────────
    lv_rows = []
    for arm in C.ARMS:
        b = BR[BR["arm"] == arm]
        ev_p = EV["primary"][EV["primary"]["arm"] == arm].groupby("eval_iter")["eval_score"].agg(["mean", "std", "count"])
        ev_h = EV["heldout"][EV["heldout"]["arm"] == arm].groupby("eval_iter")["eval_score"].agg(["mean", "std", "count"])
        eval_iters = sorted(set(ev_p.index) | set(ev_h.index))
        for ei in eval_iters:
            bb = b[b["eval_iter"] == ei]
            row = dict(arm=arm, method=C.method_of(arm), K=C.k_of(arm), eval_iter=int(ei),
                       model=f"{C.method_of(arm)}Exp3_LA{C.k_of(arm)}_{'Base' if ei == 0 else f'I{ei}'}",
                       train_iter=int(ei) + 1 if len(bb) else np.nan,
                       n_branch_points=int(len(bb)), n_convs_branched=int(bb["conversation_id"].nunique()) if len(bb) else 0,
                       mean_n_turns=float(bb["n_turns"].mean()) if len(bb) else np.nan,
                       proxy_mean=float(bb["proxy_score"].mean()) if len(bb) else np.nan,
                       proxy_sd=float(bb["proxy_score"].std(ddof=1)) if len(bb) > 1 else np.nan,
                       eval_mean_primary=float(ev_p.loc[ei, "mean"]) if ei in ev_p.index else np.nan,
                       eval_sd_primary=float(ev_p.loc[ei, "std"]) if ei in ev_p.index else np.nan,
                       n_eval_primary=int(ev_p.loc[ei, "count"]) if ei in ev_p.index else 0,
                       eval_mean_heldout=float(ev_h.loc[ei, "mean"]) if ei in ev_h.index else np.nan,
                       n_eval_heldout=int(ev_h.loc[ei, "count"]) if ei in ev_h.index else 0)
            row["gap_proxy_minus_eval_primary"] = row["proxy_mean"] - row["eval_mean_primary"]
            row["gap_proxy_minus_eval_heldout"] = row["proxy_mean"] - row["eval_mean_heldout"]
            lv_rows.append(row)
    LV = pd.DataFrame(lv_rows)
    C.save_table(LV, f"{SCRIPT}_levels",
                 caption=("Proxy-vs-eval LEVELS per arm x model state: proxy_mean = mean over that iteration's branch "
                          "points of the CHOSEN (arg-max) candidate's training-oracle score (K=0: prefix+completion; K=5: "
                          "the K-EXTENDED score, prefix+completion+5 simulated turns) — the reward the update actually "
                          "optimised, indexed by the policy that produced it (train_iter = eval_iter + 1; the final model "
                          "state, eval_iter 10 / GRPO_LA5 5, was evaluated but never trained on, hence NaN proxy). "
                          "eval_mean_primary / eval_mean_heldout = mean full-conversation Q1Q2 of the same model state "
                          "under the training oracle (gpt-4o-mini) and the held-out judge (Claude Haiku 4.5) — never "
                          "averaged. gap = proxy_mean - eval_mean (+ => the training reward reads higher than the "
                          "full-conversation eval). mean_n_turns = mean prefix length of the branch points. Iteration 0 "
                          f"= two independent base draws per method (one per K arm). {CENSOR_NOTE}"))
    # Spearman rho between proxy level and eval level over iterations, per arm and per method (pooled)
    rho_rows = []
    for arm in C.ARMS:
        d = LV[(LV["arm"] == arm) & LV["proxy_mean"].notna()]
        for j, col in (("primary", "eval_mean_primary"), ("heldout", "eval_mean_heldout")):
            dd = d[[col, "proxy_mean"]].dropna()
            rho, p = (sps.spearmanr(dd["proxy_mean"], dd[col]) if len(dd) >= 3 else (np.nan, np.nan))
            pr, pp = (sps.pearsonr(dd["proxy_mean"], dd[col]) if len(dd) >= 3 else (np.nan, np.nan))
            rho_rows.append(dict(arm=arm, eval_grader=JS[j], n_iters=len(dd), spearman_rho=float(rho), spearman_p=float(p),
                                 pearson_r=float(pr), pearson_p=float(pp),
                                 mean_gap=float((dd["proxy_mean"] - dd[col]).mean()),
                                 proxy_range=float(dd["proxy_mean"].max() - dd["proxy_mean"].min()),
                                 eval_range=float(dd[col].max() - dd[col].min())))
    RHO = pd.DataFrame(rho_rows)
    C.save_table(RHO, f"{SCRIPT}_levels_rho",
                 caption=("Across-iteration association between the training-proxy LEVEL and the full-conversation eval "
                          f"LEVEL (rows of {SCRIPT}_levels with a proxy), per arm and eval-side grader: Spearman rho and "
                          "Pearson r over n_iters model states (train_iter 1..N), mean_gap = mean(proxy - eval), and the "
                          "range each level spans over training. Descriptive: n <= 10 points per arm, no multiplicity "
                          f"correction. Proxy = training oracle by construction. {CENSOR_NOTE}"))
    for _, r in LV.iterrows():
        if np.isnan(r["proxy_mean"]):
            continue
        L.put(f"levels.{r['arm']}.eval_iter{int(r['eval_iter'])}",
              dict(proxy_mean=round(float(r["proxy_mean"]), 3), eval_mean_primary=round(float(r["eval_mean_primary"]), 3),
                   eval_mean_heldout=round(float(r["eval_mean_heldout"]), 3) if not np.isnan(r["eval_mean_heldout"]) else None,
                   gap_primary=round(float(r["gap_proxy_minus_eval_primary"]), 3), n_branch_points=int(r["n_branch_points"]),
                   mean_n_turns=round(float(r["mean_n_turns"]), 2)),
              source=f"tables/{SCRIPT}_levels.md row arm={r['arm']}, eval_iter={int(r['eval_iter'])}")
    for _, r in RHO.iterrows():
        L.put(f"levels_rho.{r['arm']}.{'primary' if r['eval_grader']==JS['primary'] else 'heldout'}",
              dict(spearman_rho=round(float(r["spearman_rho"]), 3), spearman_p=round(float(r["spearman_p"]), 4),
                   pearson_r=round(float(r["pearson_r"]), 3), n_iters=int(r["n_iters"]), mean_gap=round(float(r["mean_gap"]), 3)),
              source=f"tables/{SCRIPT}_levels_rho.md row arm={r['arm']}, eval_grader={r['eval_grader']}")

    # ── 5) figure ────────────────────────────────────────────────────────────────────────────────
    def make_fig(jkey: str, name: str):
        ab = AB[jkey]
        gl = C.JUDGE_LABEL[C.JUDGES[jkey]]
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
        for ax, m in zip(axes[:2], METHODS):
            main_iters = "1-10" if m == "PTO" else "1-5"
            for K in (0, 5):
                arm = f"{m}_LA{K}"
                d = CURVE[(CURVE["judge"] == JS[jkey]) & (CURVE["arm"] == arm) & (CURVE["iters"] == main_iters)
                          & (CURVE["n_turns"].str.isdigit())].copy()
                d["nt"] = d["n_turns"].astype(int)
                d = d.sort_values("nt")
                st = C.K_STYLE[K]
                ax.fill_between(d["nt"], d["ci_lo"], d["ci_hi"], color=pal[arm], alpha=0.18, lw=0)
                ax.plot(d["nt"], d["agreement"], color=pal[arm], ls=st["ls"], marker=st["marker"], ms=4.5, lw=1.6,
                        label=f"{arm} (iters {main_iters})")
            if m == "GRPO":   # full-support LA0 as a faint reference (its later iterations lose faithfulness)
                d = CURVE[(CURVE["judge"] == JS[jkey]) & (CURVE["arm"] == "GRPO_LA0") & (CURVE["iters"] == "1-10")
                          & (CURVE["n_turns"].str.isdigit())].copy()
                d["nt"] = d["n_turns"].astype(int)
                d = d.sort_values("nt")
                ax.plot(d["nt"], d["agreement"], color=pal["GRPO_LA0"], ls=":", lw=1.2, alpha=0.8,
                        label="GRPO_LA0 (iters 1-10)")
            ax.set_title(f"{m}: proxy vs full-conv eval", fontsize=9)
            ax.set_xlabel("prefix length n_turns (utterances)", fontsize=8)
            ax.set_ylabel("pairwise sign-agreement (0.5 = chance)", fontsize=8)
            ax.set_ylim(0.62, 1.0)
            ax.set_xlim(10, 52)
            ax.axhline(0.5, color="grey", lw=0.8, ls=":")
            ax.tick_params(labelsize=7.5)
            ax.legend(fontsize=6.8, loc="lower left", frameon=True)
            ax.grid(True, alpha=0.35)
        ax = axes[2]
        for m, off in zip(METHODS, (-0.35, 0.35)):
            d = MP[(MP["judge"] == JS[jkey]) & (MP["method"] == m) & (MP["cut"] == "train_iter_1") & (MP["n_turns"].str.isdigit())].copy()
            d["nt"] = d["n_turns"].astype(int)
            d = d.sort_values("nt")
            col = pal[f"{m}_LA0"]
            yerr = np.vstack([d["delta_K0_minus_K5"] - d["d_lo"], d["d_hi"] - d["delta_K0_minus_K5"]])
            ax.errorbar(d["nt"] + off, d["delta_K0_minus_K5"], yerr=yerr, color=col, fmt="o-" if m == "PTO" else "s-",
                        ms=4, lw=1.3, elinewidth=0.8, capsize=1.2, label=f"{m} (train_iter 1, base policy)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("matched policy: K0 − K5", fontsize=9)
        ax.set_xlabel("prefix length n_turns (utterances)", fontsize=8)
        ax.set_ylabel("Δ agreement, K0 − K5 (− = K5 more faithful)", fontsize=8)
        ax.set_xlim(10, 52)
        ax.set_ylim(-0.3, 0.3)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6.5, loc="upper left", frameon=True)
        ax.grid(True, alpha=0.35)
        fig.suptitle(f"Training-reward faithfulness — eval graded by the {gl}; proxy = training oracle",
                     fontsize=9, y=1.02)
        return C.save_fig(fig, name)

    p1 = make_fig("primary", f"{SCRIPT}_fig")
    p2 = make_fig("heldout", f"{SCRIPT}_fig_heldout")
    L.put("figures", {"fig": str(p1), "fig_heldout": str(p2)},
          source="figures/reward_faithfulness_fig.png / _fig_heldout.png",
          note=("Panels a,b: agreement vs n_turns per method, K=0 solid/circle, K=5 dashed/square, ribbons = 95% cluster-"
                "bootstrap CI, pooled over iterations (GRPO_LA5 censored at 5). Panel c: matched-policy (train_iter 1) "
                "K0-K5 difference per bin with CI; negative = look-ahead more faithful."))

    # ── run-level numbers into ledger ────────────────────────────────────────────────────────────
    L.put("bootstrap", dict(B=N_BOOT, seed=SEED, cluster="conversation within (arm, eval_iter)"), source="script constants")
    L.put("caveats", [
        "proxy_score is the training oracle's score by construction; it cannot be re-graded, so the held-out-judge tables change only the EVAL side.",
        "PTO greedy trunks share only the first MCL=12 utterances with the eval conversation; beyond n_turns=12 the PTO prefix follows the best-of-M trunk, so PTO's curve mixes cut length with trunk divergence (GRPO prefixes are exact slices).",
        "GRPO branch rows include the policy drifting within an iteration (2 epochs) and ~3-10% eval-split groups scored at iteration end; PTO rows come from the frozen iter-start policy.",
        "GRPO_LA5 is right-censored at train_iter 5; K reads for GRPO use the iters 1-5 series / cuts.",
        "Pairs within a cell are not independent (each conversation enters n-1 pairs) — CIs come from the conversation-level cluster bootstrap, not from n_pairs.",
        "Wilcoxon over n_turns bins treats correlated bins as observations — descriptive only.",
        "GRPO_LA5 iteration 1 captured only its second epoch (the run resumed after a crash and the recorder flushes once per iteration).",
    ], source="script")
    L.put("n_branch_rows", int(len(BR)), source="training.load_branch_reliability(which='chosen') row count")
    L.put("n_branch_rows_joined_primary", int(len(DF["primary"])), source="inner join to primary eval Q1Q2 on (arm, eval_iter, conversation_id)")
    L.put("n_branch_rows_joined_heldout", int(len(DF["heldout"])), source="inner join to held-out eval Q1Q2")
    L.save()
    print("done")


if __name__ == "__main__":
    main()
