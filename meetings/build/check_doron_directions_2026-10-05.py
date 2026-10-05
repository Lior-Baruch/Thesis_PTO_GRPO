"""
check_doron_directions_2026-10-05.py — Doron's 2026-10 "win/lose direction" deck re-run with gte on our
round data, plus the leaked-marker / degenerate-text check (Lior's request, 2026-10-05).

Doron's quantity: per arm x training iteration, the mean over rounds of embedding(best) − embedding(worst),
renormalised. He compares these directions across iterations (stability) and between the arms, reads
them by projecting a pool of frequent real sentences onto them, and reads K0 − K5 the same way.

Here the same quantities are computed on the rounds the 2026-09-29 analysis used (`train`, not all tied;
best and worst average over ties), with that analysis's cached gte-base embeddings, so no candidate is
re-embedded. Additions:
  * every quantity is repeated with leaked-chat-marker candidates removed, and with those AND
    degenerate-text candidates removed (best and worst are re-chosen among the remaining candidates);
  * every cosine has a noise-corrected version beside it. Patients are split in half 50 times;
    corrected cos = mean cos(half A of one direction, half B of the other) / sqrt(rel_1 * rel_2), where
    rel is each direction's own split-half cosine. About 1 = the same direction once noise is removed;
  * how often the best and the worst candidate carry each artefact;
  * which of the 2026-09-29 text features (fixed before Doron's deck existed) the K0 − K5 difference
    lines up with, beside the counts from Doron's own post-hoc keyword lists.

Usage (repo root, repo .venv; embeds only the ~4.5k pool sentences, <= ~1.5 GB VRAM):
    .venv/Scripts/python.exe meetings/build/check_doron_directions_2026-10-05.py [--share <dir>] [--out <dir>]
"""

import argparse
import collections
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd
import torch  # noqa: F401  # before transformers (local sm_120 import order)
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
_spec = importlib.util.spec_from_file_location("round_directions",
                                               os.path.join(HERE, "analyze_grpo_round_directions_2026-09-29.py"))
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

DEFAULT_OUT = os.path.join(REPO, "meetings", "2026-10-05_doron_direction_check")
GTE_CACHE = os.path.join(rd.DEFAULT_OUT, ".emb_cache")   # the 2026-09-29 candidate embeddings
ARMS, ITERS = rd.ARMS, rd.ITERS
CELLS = [(a, i) for a in ARMS for i in ITERS]
LAB = rd.LAB
POOL_SIZE, MIN_COUNT, MIN_WORDS, MAX_WORDS = 4488, 3, 4, 10   # Doron's pool recipe
N_SPLITS, SEED, TOP, TOP_KW = 50, 0, 3, 10
LEAK = rd.FEATURES["leaked chat marker"]
VARIANTS = {"all": "all candidates",
            "no_leak": "no leaked chat marker",
            "clean": "no leaked chat marker and no degenerate text"}
# Doron's keyword lists, as written on his "Confirmed by keyword count" slide (chosen after seeing his sentences)
DORON_K0 = re.compile(r"proud|thrilled|grateful|vulnerab|trust|courage|here for you", re.I)
DORON_K5 = re.compile(r"instead of|what if|different (perspective|approach)|break this down|interesting observation", re.I)


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


# ── sentence pool ─────────────────────────────────────────────────────────────────────────────
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_pool(texts):
    c = collections.Counter()
    for t in texts:
        for s in _SPLIT.split(t):
            s = s.strip()
            if MIN_WORDS <= len(s.split()) <= MAX_WORDS:
                c[s] += 1
    kept = [(s, n) for s, n in c.most_common() if n >= MIN_COUNT][:POOL_SIZE]
    pool = pd.DataFrame(kept, columns=["sentence", "count"])
    pool["has_marker"] = pool.sentence.map(lambda s: bool(LEAK.search(s)))
    return pool, len(c)


# ── directions ────────────────────────────────────────────────────────────────────────────────
def round_deltas(D, keep):
    """Per round: mean embedding of the best kept candidate(s) − that of the worst kept. A round counts
    only if its kept candidates have at least two different rewards."""
    R, n, cr = len(D.rounds), len(D.r), D.cand_round
    starts = D.rounds.start.to_numpy()
    rmax = np.maximum.reduceat(np.where(keep, D.r, -np.inf), starts)
    rmin = np.minimum.reduceat(np.where(keep, D.r, np.inf), starts)
    valid = rmax > rmin
    best = keep & valid[cr] & (D.r == rmax[cr])
    worst = keep & valid[cr] & (D.r == rmin[cr])
    w = np.zeros(n)
    nb, nw = np.bincount(cr[best], minlength=R), np.bincount(cr[worst], minlength=R)
    w[best], w[worst] = 1.0 / nb[cr[best]], -1.0 / nw[cr[worst]]
    nz = np.flatnonzero(w)
    M = sparse.csr_matrix((w[nz], (cr[nz], nz)), shape=(R, n))
    return np.asarray(M @ D.E), valid


class Directions:
    """Full-sample direction per cell and its 50 split-half directions (same patient splits everywhere)."""

    def __init__(self, D, delta, valid, splits):
        P = int(D.rounds.patient.max()) + 1
        self.full, self.halves, self.n_rounds = {}, {}, {}
        arm, it, pat = D.rounds.arm.to_numpy(), D.rounds.iteration.to_numpy(), D.rounds.patient.to_numpy()
        for cell in CELLS:
            idx = np.flatnonzero(valid & (arm == cell[0]) & (it == cell[1]))
            S = np.zeros((P, delta.shape[1]))
            np.add.at(S, pat[idx], delta[idx])
            self.n_rounds[cell] = len(idx)
            self.full[cell] = unit(S.sum(0))
            self.halves[cell] = np.stack([unit(np.stack([S[a].sum(0), S[b].sum(0)])) for a, b in splits])  # (N, 2, d)

    def rel(self, c):
        h = self.halves[c]
        return float(np.mean(np.einsum("nd,nd->n", h[:, 0], h[:, 1])))

    def cos(self, c1, c2):
        return float(self.full[c1] @ self.full[c2])

    def cos_corrected(self, c1, c2):
        h1, h2 = self.halves[c1], self.halves[c2]
        x = 0.5 * (np.einsum("nd,nd->n", h1[:, 0], h2[:, 1]) + np.einsum("nd,nd->n", h1[:, 1], h2[:, 0]))
        r1, r2 = self.rel(c1), self.rel(c2)
        return float(np.mean(x) / np.sqrt(r1 * r2)) if r1 > 0 and r2 > 0 else np.nan


# ── output helpers ────────────────────────────────────────────────────────────────────────────
def md(s):
    return s.replace("|", "\\|").replace("\n", " ")


def fmt_sents(sents, flags):
    return "<br>".join(f"“{md(s)}”" + (" ⚑" if f else "") for s, f in zip(sents, flags))


def top_bottom(emb, v, k):
    p = emb @ v
    o = np.argsort(-p)
    return o[:k], o[::-1][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", default=rd.DEFAULT_SHARE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    tab = os.path.join(out, "tables")
    os.makedirs(tab, exist_ok=True)

    print("loading rounds (cached gte embeddings) ...", flush=True)
    D = rd.Data(a.share, GTE_CACHE)
    R, cr = len(D.rounds), D.cand_round
    leak = np.array([bool(LEAK.search(t)) for t in D.text])
    degen = D.degenerate.astype(bool)
    F = rd.text_features(D.text, D.length, D.degenerate)
    cand_arm = D.rounds.arm.to_numpy()[cr]
    cand_it = D.rounds.iteration.to_numpy()[cr]
    keeps = {"all": np.ones(len(D.r), bool), "no_leak": ~leak, "clean": ~leak & ~degen}

    # ── 1. how often the best / worst candidate carries each artefact (all rounds) ──
    rows = []
    for arm in ARMS:
        for it in ITERS + ["all"]:
            m = (cand_arm == arm) & ((cand_it == it) if it != "all" else True)
            rr = np.flatnonzero((D.rounds.arm == arm).to_numpy() & ((D.rounds.iteration == it).to_numpy() if it != "all" else True))
            sub = np.isin(cr, rr)
            row = dict(arm=arm, iteration=it, candidates=int(m.sum()))
            for name, x in (("leak", leak), ("degenerate", degen), ("cut_off", F["cut off mid-sentence"].to_numpy().astype(bool))):
                row[f"{name}_share"] = float(x[m].mean())
                row[f"{name}_among_best"] = float(x[m & D.is_best].mean())
                row[f"{name}_among_worst"] = float(x[m & D.is_worst].mean())
                row[f"{name}_reward_corr_within_round"] = rd.within_corr(x[sub].astype(float), D.r[sub], cr[sub], R)
            rows.append(row)
    art = pd.DataFrame(rows)
    art.round(4).to_csv(os.path.join(tab, "artefact_rates.csv"), index=False)
    print(art[art.iteration == "all"].round(3).T.to_string(), flush=True)

    # ── 2. directions per variant ──
    rng = np.random.default_rng(SEED)
    P = int(D.rounds.patient.max()) + 1
    pats = np.unique(D.rounds.patient)
    splits = []
    for _ in range(N_SPLITS):
        p = rng.permutation(pats)
        A, B = np.zeros(P, bool), np.zeros(P, bool)
        A[p[: len(p) // 2]], B[p[len(p) // 2:]] = True, True
        splits.append((A, B))
    dirs = {}
    for v in VARIANTS:
        delta, valid = round_deltas(D, keeps[v])
        dirs[v] = Directions(D, delta, valid, splits)
        print(f"  {v}: {int(valid.sum()):,} rounds", flush=True)

    rows = []
    for v, Dv in dirs.items():
        for arm in ARMS:
            for it in ITERS:
                c = (arm, it)
                row = dict(variant=v, arm=arm, iteration=it, n_rounds=Dv.n_rounds[c], reliability=Dv.rel(c))
                if it > 1:
                    row["stability_cos"] = Dv.cos(c, (arm, it - 1))
                    row["stability_corrected"] = Dv.cos_corrected(c, (arm, it - 1))
                if arm == "GRPO_K0":
                    row["k0_vs_k5_cos"] = Dv.cos(c, ("GRPO_K5", it))
                    row["k0_vs_k5_corrected"] = Dv.cos_corrected(c, ("GRPO_K5", it))
                row["vs_all_cos"] = Dv.full[c] @ dirs["all"].full[c]
                rows.append(row)
    cos = pd.DataFrame(rows)
    cos.round(4).to_csv(os.path.join(tab, "cosines.csv"), index=False)

    # ── 3. sentence pool and projections ──
    print("building the sentence pool ...", flush=True)
    pool, n_len = sentence_pool(D.text)
    pool.to_csv(os.path.join(tab, "sentence_pool.csv"), index=False)
    print(f"  {n_len:,} distinct 4–10-word sentences; pool {len(pool):,} "
          f"({pool.has_marker.mean():.1%} carry a chat marker)", flush=True)
    emb = rd.embed(pool.sentence.tolist(), os.path.join(out, ".emb_cache"))
    emb = unit(emb)
    S, flag = pool.sentence.to_numpy(), pool.has_marker.to_numpy()

    lines = ["# Top win / lose sentences per arm and iteration\n",
             "The pool sentences that project highest (win) and lowest (lose) on each arm × iteration direction",
             "(mean embedding(best) − embedding(worst)). ⚑ = the sentence contains a leaked chat marker.\n"]
    marker_rows = []
    for v, Dv in dirs.items():
        lines += [f"\n## {VARIANTS[v]}\n", "| iter | arm | win | lose |", "|---|---|---|---|"]
        for it in ITERS:
            for arm in ARMS:
                hi, lo = top_bottom(emb, Dv.full[(arm, it)], max(TOP, TOP_KW))
                lines.append(f"| {it} | {LAB[arm]} | {fmt_sents(S[hi[:TOP]], flag[hi[:TOP]])} | {fmt_sents(S[lo[:TOP]], flag[lo[:TOP]])} |")
                marker_rows.append(dict(variant=v, arm=arm, iteration=it, direction="arm",
                                        marker_share_top10_win=float(flag[hi[:TOP_KW]].mean()),
                                        marker_share_top10_lose=float(flag[lo[:TOP_KW]].mean())))
    open(os.path.join(tab, "top_sentences.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ── 4. K0 − K5 ──
    lines = ["# Sentences K=0's direction favours more than K=5's, and the reverse\n",
             "Pool sentences projected on (K=0 direction − K=5 direction), same iteration. ⚑ = leaked chat marker.\n"]
    kw_rows, feat_rows = [], []
    for v, Dv in dirs.items():
        lines += [f"\n## {VARIANTS[v]}\n", "| iter | cos(K0, K5) | more K=0 | more K=5 |", "|---|---|---|---|"]
        for it in ITERS:
            d = Dv.full[("GRPO_K0", it)] - Dv.full[("GRPO_K5", it)]
            hi, lo = top_bottom(emb, d, TOP_KW)
            lines.append(f"| {it} | {Dv.cos(('GRPO_K0', it), ('GRPO_K5', it)):.2f} | "
                         f"{fmt_sents(S[hi[:TOP]], flag[hi[:TOP]])} | {fmt_sents(S[lo[:TOP]], flag[lo[:TOP]])} |")
            for side, idx in (("more_K0", hi), ("more_K5", lo)):
                kw_rows.append(dict(variant=v, iteration=it, side=side,
                                    doron_K0_list_hits=sum(bool(DORON_K0.search(s)) for s in S[idx]),
                                    doron_K5_list_hits=sum(bool(DORON_K5.search(s)) for s in S[idx]),
                                    marker_sentences=int(flag[idx].sum())))
            # which 2026-09-29 text features the difference lines up with, WITHIN each arm's candidates
            du = unit(d)
            for arm in ARMS:
                m = keeps[v] & (cand_arm == arm) & (cand_it == it)
                proj = D.E[m] @ du
                for name in F.columns:
                    x = F[name].to_numpy()[m]
                    r = float(np.corrcoef(x, proj)[0, 1]) if x.std() > 0 else np.nan
                    feat_rows.append(dict(variant=v, iteration=it, arm=arm, feature=name, corr=r, share_or_mean=float(x.mean())))
    open(os.path.join(tab, "k0_minus_k5_sentences.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    kw = pd.DataFrame(kw_rows)
    kw.to_csv(os.path.join(tab, "doron_keyword_counts.csv"), index=False)
    feat = pd.DataFrame(feat_rows)
    feat.round(4).to_csv(os.path.join(tab, "k0_minus_k5_features.csv"), index=False)
    pd.DataFrame(marker_rows).round(3).to_csv(os.path.join(tab, "marker_share_top_sentences.csv"), index=False)

    summary = dict(
        rounds={v: int(sum(Dv.n_rounds.values())) for v, Dv in dirs.items()},
        candidates=int(len(D.r)), pool_size=int(len(pool)), distinct_4_10_word_sentences=int(n_len),
        pool_marker_share=float(pool.has_marker.mean()),
        keyword_totals=kw.groupby(["variant", "side"])[["doron_K0_list_hits", "doron_K5_list_hits", "marker_sentences"]].sum()
                         .reset_index().to_dict("records"),
        feature_corr_mean=feat.groupby(["variant", "feature"])["corr"].mean().round(3).unstack(0).to_dict(),
    )
    json.dump(summary, open(os.path.join(tab, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:4000])
    print("done:", out)


if __name__ == "__main__":
    main()
