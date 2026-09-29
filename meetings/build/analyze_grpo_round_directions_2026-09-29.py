"""
analyze_grpo_round_directions_2026-09-29.py — which direction in reply-embedding space each GRPO round
rewards, estimated two ways, for both GRPO arms (Doron's request, 2026-09-29).

Reads ONLY the Doron share (``grpo_groups/``), so it reruns on his copy. Every reply is embedded with
``Alibaba-NLP/gte-base-en-v1.5`` (mean-pooled, unit norm). A round's 8 candidates share one
conversation so far, so everything below is WITHIN a round: what separates the candidates, never what
separates rounds.

**Analysis 1 — best vs worst.** Each round contributes one pair: the mean embedding of its top-reward
candidate(s) minus that of its bottom-reward candidate(s), Δ. The direction w is a Bradley-Terry probe:
P(best beats worst) = σ(w·Δ), an L2-regularised logistic regression on ±Δ without intercept. The
simple mean of Δ is reported beside it as the no-model baseline.

**Analysis 2 — all 8.** Every candidate contributes, weighted by its GRPO advantage
a = (reward − round mean) / round std. The direction is the ridge regression of a on the
round-centred embedding (the linear map from a reply's embedding to the advantage GRPO gives it). The
advantage-weighted mean embedding (GRPO's own weighting, no model) is the baseline.

Both directions are scored on the same held-out rounds (5 folds by patient — the same folds for every
arm and iteration, so no patient is ever on both sides) with three per-round scores:
  * best-vs-worst accuracy — does the best candidate project above the worst;
  * all-pairs accuracy — over every pair of the round's candidates with different rewards;
  * within-round Spearman — projection vs reward over all the round's candidates.
Beside every score: the same score for reply length alone. 95% CIs resample patients.

Also: split-half reliability of each direction; a transfer matrix (fit on one arm x iteration, test on
every other); accuracy by the round's reward gap; and which interpretable text features each direction
and the reward itself favour.

Usage (repo root, repo .venv; only embeds texts not already in the cache, at <= ~1.5 GB VRAM):
    .venv/Scripts/python.exe meetings/build/analyze_grpo_round_directions_2026-09-29.py \
        [--share <Exp3_share_for_Doron>] [--out <dir>]
"""

import argparse
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
import torch                                              # before transformers / sklearn (local sm_120)
from transformers import AutoModel, AutoTokenizer
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
import seaborn as sns                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SHARE = r"G:\My Drive\Thesis_PTO_GRPO\Exp3_share_for_Doron"
DEFAULT_OUT = os.path.join(REPO, "meetings", "2026-09-29_doron_grpo_groups", "directions")

MODEL = "Alibaba-NLP/gte-base-en-v1.5"
ARMS = ["GRPO_K0", "GRPO_K5"]
ITERS = list(range(1, 11))
CELLS = [(a, i) for a in ARMS for i in ITERS]
COL = {"GRPO_K0": "#d55e00", "GRPO_K5": "#e69f00"}       # the paper's K=0 / K=5 colours
MARK = {"GRPO_K0": "o", "GRPO_K5": "s"}
LAB = {"GRPO_K0": "K=0", "GRPO_K5": "K=5"}
GREY, LIGHT = "#6b6b6b", "#b0b0b0"
N_FOLDS, N_BOOT, N_SPLITS, SEED = 5, 1000, 20, 0
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]                  # Bradley-Terry (logistic) inverse penalty
ALPHA_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]          # ridge penalty

# Interpretable text features. PRAISE / AFFIRM are copied from the Exp3 EDA
# (eda_analysis/constants.py RE_EFFUSIVE / RE_AFFIRM) so the two agree.
FEATURES = {
    "length (log chars)": None,
    "praise phrase": re.compile(r"\bi'?m so proud|proud of you|inspiration to me|you got this|beautiful|beacon|"
                                r"shining|warrior|hero of your|you are a (light|beacon)", re.I),
    "'you are ...' affirmation": re.compile(r"\byou are\b|\byou're (worthy|enough|strong|powerful|brave|amazing|a )", re.I),
    "asks a question": re.compile(r"\?"),
    "reflection opener": re.compile(r"\b(it sounds like|it seems like|sounds like you|you('re| are) feeling|"
                                    r"what i('m| am) hearing|so you('re| are))\b", re.I),
    "advice / tips": re.compile(r"\b(you (should|could|might|can) (try|consider|start)|try to|i (suggest|recommend)|"
                                r"here are|tips?|strateg(y|ies)|steps?)\b", re.I),
    "numbered or bulleted list": re.compile(r"(^|\n)\s*(\d+[.)]|[-*\u2022])\s"),
    "'we' / 'together'": re.compile(r"\b(we|let's|together)\b", re.I),
    "cut off mid-sentence": None,
    "degenerate text": None,
    "leaked chat marker": re.compile(r"<\|?im_"),
}


# ── embedding (cache keyed by text, so only new texts are embedded) ───────────────────────────
def load_gte():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
    # transformers 5 leaves this remote code's non-persistent buffers uninitialised; rebuild them.
    m.embeddings.register_buffer("position_ids", torch.arange(m.config.max_position_embeddings), persistent=False)
    m.embeddings._init_rope(m.config)
    return tok, m.cuda()


def _encode(texts: list, max_batch: int = 128, token_budget: int = 8192) -> np.ndarray:
    """VRAM: plain attention grows with batch x length^2, and on the 12 GB local card an over-budget
    request reboots the PC instead of raising. So batches are sized by a token budget and the allocator
    is capped, making an overrun raise OutOfMemoryError instead."""
    torch.cuda.set_per_process_memory_fraction(0.5)
    tok, m = load_gte()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 768), dtype=np.float16)
    i = 0
    while i < len(order):                                 # chars/3 over-estimates tokens: a safe size
        n = max(1, min(max_batch, token_budget // max(1, min(512, len(texts[order[min(i + max_batch, len(order)) - 1]]) // 3))))
        idx = order[i:i + n]
        x = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = m(**x).last_hidden_state.float()
        mask = x["attention_mask"].unsqueeze(-1).float()
        out[idx] = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1).cpu().numpy()
        i += n
    del m
    torch.cuda.empty_cache()
    return out


def embed(texts: list, cache_dir: str) -> np.ndarray:
    tpath, epath = os.path.join(cache_dir, "gte_texts.json.gz"), os.path.join(cache_dir, "gte_emb.npy")
    known, E = [], np.zeros((0, 768), dtype=np.float16)
    if os.path.exists(tpath) and os.path.exists(epath):
        known = json.load(gzip.open(tpath, "rt", encoding="utf-8"))
        E = np.load(epath)
    pos = {t: k for k, t in enumerate(known)}
    missing = sorted({t for t in texts if t not in pos})
    if missing:
        print(f"  embedding {len(missing):,} new texts ...", flush=True)
        E = np.vstack([E, _encode(missing)])
        known += missing
        pos.update({t: len(known) - len(missing) + k for k, t in enumerate(missing)})
        os.makedirs(cache_dir, exist_ok=True)
        np.save(epath, E)
        with gzip.open(tpath, "wt", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
    return E[[pos[t] for t in texts]].astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── degenerate text ───────────────────────────────────────────────────────────────────────────
_NON_ASCII = re.compile("[^\x00-\x7f‘’“”–—…•]")   # curly quotes etc. are fine
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def degenerate_text(texts: list, min_docs: int = 200) -> np.ndarray:
    """1.0 for an empty reply, a reply that is > 3% non-ASCII characters, or one where fewer than 70% of
    the words are in the corpus vocabulary (words used in >= ``min_docs`` candidates) - i.e. word salad.
    Deliberately lenient: text that drifts into salad late in the reply can still pass."""
    words = [[w.lower() for w in _WORD.findall(t)] for t in texts]
    df = {}
    for ws in words:
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    vocab = {w for w, c in df.items() if c >= min_docs}
    out = np.zeros(len(texts))
    for k, (t, ws) in enumerate(zip(texts, words)):
        if not t.strip() or len(_NON_ASCII.findall(t)) / len(t) > 0.03 or \
                (ws and sum(w in vocab for w in ws) / len(ws) < 0.7):
            out[k] = 1.0
    return out


# ── data ──────────────────────────────────────────────────────────────────────────────────────
class Data:
    """Candidate-level and round-level arrays; a round's candidates are contiguous."""

    def __init__(self, share: str, cache: str):
        cand_rows, round_rows = [], []
        for arm in ARMS:
            for it in ITERS:
                for line in gzip.open(os.path.join(share, "grpo_groups", arm, f"iter_{it:02d}.jsonl.gz"), "rt", encoding="utf-8"):
                    r = json.loads(line)
                    if r["split"] != "train":
                        continue
                    cs = [c for c in r["candidates"] if c["reward"] is not None]
                    if len({c["reward"] for c in cs}) < 2:          # no winner and no loser: no signal
                        continue
                    start = len(cand_rows)
                    cand_rows += [(c["text"], float(c["reward"])) for c in cs]
                    round_rows.append((arm, it, r["round"], r["epoch"], r["patient_id"], r["turn"], start, len(cand_rows)))
        self.text = [t for t, _ in cand_rows]
        self.r = np.array([x for _, x in cand_rows])
        self.rounds = pd.DataFrame(round_rows, columns=["arm", "iteration", "round", "epoch", "patient", "turn", "start", "end"])
        self.cand_round = np.repeat(np.arange(len(self.rounds)), self.rounds.end - self.rounds.start)
        print(f"  {len(self.rounds):,} rounds, {len(self.text):,} candidates", flush=True)
        self.E = embed(self.text, cache)
        self.length = np.array([len(t) for t in self.text], dtype=float)
        self.degenerate = degenerate_text(self.text)
        R = len(self.rounds)
        self.rounds["clean"] = np.bincount(self.cand_round, weights=self.degenerate, minlength=R) == 0
        print(f"  degenerate text: {self.degenerate.mean():.1%} of candidates; "
              f"{self.rounds.clean.mean():.1%} of rounds have none", flush=True)
        self._round_stats()
        self._pairs()

    def _round_stats(self):
        R, n = len(self.rounds), len(self.r)
        cnt = np.bincount(self.cand_round, minlength=R)
        mean = np.bincount(self.cand_round, weights=self.r, minlength=R) / cnt
        dev = self.r - mean[self.cand_round]
        std = np.sqrt(np.bincount(self.cand_round, weights=dev ** 2, minlength=R) / (cnt - 1))
        self.adv = dev / (std[self.cand_round] + 1e-4)       # GRPO's group-relative advantage
        rmax = np.maximum.reduceat(self.r, self.rounds.start.to_numpy())
        rmin = np.minimum.reduceat(self.r, self.rounds.start.to_numpy())
        is_best = self.r == rmax[self.cand_round]
        is_worst = self.r == rmin[self.cand_round]
        emb_mean = np.zeros((R, 768), dtype=np.float32)
        np.add.at(emb_mean, self.cand_round, self.E)
        emb_mean /= cnt[:, None]
        self.Xc = self.E - emb_mean[self.cand_round]      # round-centred embeddings
        best = np.zeros((R, 768), dtype=np.float32)
        worst = np.zeros((R, 768), dtype=np.float32)
        np.add.at(best, self.cand_round[is_best], self.E[is_best])
        np.add.at(worst, self.cand_round[is_worst], self.E[is_worst])
        best /= np.bincount(self.cand_round[is_best], minlength=R)[:, None]
        worst /= np.bincount(self.cand_round[is_worst], minlength=R)[:, None]
        self.delta = best - worst
        self.is_best, self.is_worst = is_best, is_worst
        self.rounds["gap"] = rmax - rmin
        self.rounds["spread"] = std
        lb = np.bincount(self.cand_round[is_best], weights=self.length[is_best], minlength=R) / np.bincount(self.cand_round[is_best], minlength=R)
        lw = np.bincount(self.cand_round[is_worst], weights=self.length[is_worst], minlength=R) / np.bincount(self.cand_round[is_worst], minlength=R)
        self.len_delta = lb - lw

    def _pairs(self):
        pi, pj = [], []
        for s, e in zip(self.rounds.start, self.rounds.end):
            a, b = np.triu_indices(e - s, 1)
            pi.append(a + s)
            pj.append(b + s)
        pi, pj = np.concatenate(pi), np.concatenate(pj)
        keep = self.r[pi] != self.r[pj]
        self.pi, self.pj = pi[keep], pj[keep]
        self.pair_round = self.cand_round[self.pi]
        self.pair_sign = np.sign(self.r[self.pi] - self.r[self.pj])


# ── fitting ───────────────────────────────────────────────────────────────────────────────────
def fit_bt(delta: np.ndarray, C: float) -> np.ndarray:
    X = np.vstack([delta, -delta])
    y = np.r_[np.ones(len(delta)), np.zeros(len(delta))]
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000)
    m.fit(X, y)
    return unit(m.coef_[0])


def fit_meandiff(delta):
    return unit(delta.mean(0))


def fit_ridge(Xc, adv, alpha):
    return unit(np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ adv))


def fit_advmean(Xc, adv):
    return unit(Xc.T @ adv)


# ── per-round scores ──────────────────────────────────────────────────────────────────────────
def round_scores(D: Data, p: np.ndarray, rounds: np.ndarray) -> pd.DataFrame:
    """Per-round best-vs-worst, all-pairs accuracy and within-round Spearman of the candidate scores ``p``."""
    R = len(D.rounds)
    cnt_b = np.bincount(D.cand_round[D.is_best], minlength=R)
    cnt_w = np.bincount(D.cand_round[D.is_worst], minlength=R)
    pb = np.bincount(D.cand_round[D.is_best], weights=p[D.is_best], minlength=R) / cnt_b
    pw = np.bincount(D.cand_round[D.is_worst], weights=p[D.is_worst], minlength=R) / cnt_w
    bw = np.where(pb > pw, 1.0, np.where(pb == pw, 0.5, 0.0))
    s = np.sign(p[D.pi] - p[D.pj])
    agree = np.where(s == 0, 0.5, (s == D.pair_sign).astype(float))
    npair = np.bincount(D.pair_round, minlength=R)
    ap = np.bincount(D.pair_round, weights=agree, minlength=R) / np.maximum(npair, 1)
    rho = np.full(R, np.nan)
    for k in rounds:
        a, b = D.rounds.start.iat[k], D.rounds.end.iat[k]
        x, y = rankdata(p[a:b]), rankdata(D.r[a:b])
        if x.std() > 0 and y.std() > 0:
            rho[k] = np.corrcoef(x, y)[0, 1]
    return pd.DataFrame({"round_idx": rounds, "patient": D.rounds.patient.to_numpy()[rounds],
                         "clean": D.rounds.clean.to_numpy()[rounds],
                         "bw": bw[rounds], "allpairs": ap[rounds], "rho": rho[rounds]})


def cluster_ci(df, col, rng):
    per = df.dropna(subset=[col]).groupby("patient")[col].agg(["sum", "count"])
    s, n = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── the analysis ──────────────────────────────────────────────────────────────────────────────
class Analysis:
    def __init__(self, D: Data):
        self.D = D
        rng = np.random.default_rng(SEED)
        self.fold_of_patient = dict(zip(rng.permutation(96), np.arange(96) % N_FOLDS))
        rd = D.rounds
        self.fold = rd.patient.map(self.fold_of_patient).to_numpy()
        self.cell_rounds = {c: np.flatnonzero((rd.arm == c[0]).to_numpy() & (rd.iteration == c[1]).to_numpy()) for c in CELLS}

    def cands_of(self, rounds):
        D = self.D
        return np.concatenate([np.arange(D.rounds.start.iat[k], D.rounds.end.iat[k]) for k in rounds])

    def fit(self, method, rounds, hp=None):
        D = self.D
        if method == "bt":
            return fit_bt(D.delta[rounds], hp)
        if method == "meandiff":
            return fit_meandiff(D.delta[rounds])
        c = self.cands_of(rounds)
        if method == "ridge":
            return fit_ridge(D.Xc[c], D.adv[c], hp)
        if method == "advmean":
            return fit_advmean(D.Xc[c], D.adv[c])
        raise ValueError(method)

    def cv(self, method, cell, hp=None):
        """Held-out per-round scores for one cell; also returns the per-fold directions."""
        rounds = self.cell_rounds[cell]
        parts, ws = [], {}
        for f in range(N_FOLDS):
            tr, te = rounds[self.fold[rounds] != f], rounds[self.fold[rounds] == f]
            if len(tr) == 0 or len(te) == 0:
                continue
            ws[f] = w = self.fit(method, tr, hp)
            parts.append(round_scores(self.D, self.D.E @ w, te))
        return pd.concat(parts), ws

    def select(self, method, grid, score):
        means = {}
        for hp in grid:
            means[hp] = float(np.mean([self.cv(method, c, hp)[0][score].mean() for c in CELLS]))
            print(f"    {method} hp={hp:g}: mean held-out {score} {means[hp]:.4f}", flush=True)
        return max(means, key=means.get), means

    def split_half(self, method, cell, hp, rng):
        rounds = self.cell_rounds[cell]
        pats = np.unique(self.D.rounds.patient.to_numpy()[rounds])
        vals = []
        for _ in range(N_SPLITS):
            half = set(rng.permutation(pats)[: len(pats) // 2])
            inA = np.array([p in half for p in self.D.rounds.patient.to_numpy()[rounds]])
            vals.append(float(self.fit(method, rounds[inA], hp) @ self.fit(method, rounds[~inA], hp)))
        return float(np.mean(vals))


# ── figures ───────────────────────────────────────────────────────────────────────────────────
def style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _chance(ax, y):
    ax.axhline(y, color="#444444", lw=1, ls=":")


def fig_analysis(res, gapq, out, main, base, score, ylab, gap_lab, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, arm in zip(axes[:2], ARMS):
        g = res[res.arm == arm]
        ax.fill_between(g.iteration, g[f"{score}_{main}_lo"], g[f"{score}_{main}_hi"], color=COL[arm], alpha=0.2, lw=0)
        ax.plot(g.iteration, g[f"{score}_{main}"], color=COL[arm], marker=MARK[arm], ms=4.5, lw=2, label=title[0])
        ax.plot(g.iteration, g[f"{score}_{main}_clean"], color=COL[arm], marker=MARK[arm], mfc="white", ms=4, lw=1.1,
                ls="--", label="same, scored on rounds without degenerate text")
        ax.plot(g.iteration, g[f"{score}_{base}"], color=GREY, ls="--", lw=1.3, marker=".", label=title[1])
        ax.plot(g.iteration, g[f"{score}_length"], color=LIGHT, ls=":", lw=1.5, marker=".", label="reply length alone")
        _chance(ax, 0.5 if score != "rho" else 0.0)
        ax.set_xticks(ITERS)
        ax.set_xlabel("training iteration")
        ax.set_title(f"{LAB[arm]}")
    axes[0].set_ylabel(ylab)
    axes[1].sharey(axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    ax = axes[2]
    for arm in ARMS:
        g = gapq[gapq.arm == arm]
        ax.plot(g.quintile, g[score], color=COL[arm], marker=MARK[arm], ms=4.5, lw=2, label=LAB[arm])
    _chance(ax, 0.5 if score != "rho" else 0.0)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel(gap_lab)
    ax.set_title("by how different the round's rewards are")
    ax.set_ylabel(ylab.split("\n")[0])
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, [l.replace("same, scored", "the same direction, scored") for l in labels], loc="lower left",
               bbox_to_anchor=(0.05, 0.0), ncol=4, frameon=False, fontsize=8.5,
               title="line style (colour = arm):", title_fontsize=8.5)
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


def fig_compare(cmp_df, out):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    panels = [("bw", "held-out: best projects above worst", 0.5, "(a) picking the winner over the loser"),
              ("rho", "held-out: within-round Spearman", 0.0, "(b) ranking all 8 candidates"),
              ("reliability", "split-half reliability (cosine)", None, "(c) how stable the direction is")]
    for ax, (col, ylab, ref, title) in zip(axes.flat[:3], panels):
        for arm in ARMS:
            g = cmp_df[cmp_df.arm == arm]
            ax.plot(g.iteration, g[f"{col}_bt"], color=COL[arm], marker=MARK[arm], mfc="white", ms=4.5, lw=1.4, ls="--",
                    label=f"{LAB[arm]}, best vs worst")
            ax.plot(g.iteration, g[f"{col}_ridge"], color=COL[arm], marker=MARK[arm], ms=4.5, lw=2, label=f"{LAB[arm]}, all 8")
        if ref is not None:
            _chance(ax, ref)
        ax.set_xticks(ITERS)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ax = axes[1, 1]
    for arm in ARMS:
        g = cmp_df[cmp_df.arm == arm]
        ax.plot(g.iteration, g.cos_bt_ridge, color=COL[arm], marker=MARK[arm], ms=4.5, lw=2, label=LAB[arm])
    _chance(ax, 0.0)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xticks(ITERS)
    ax.set_ylabel("cosine")
    ax.set_title("(d) best-vs-worst direction vs all-8 direction")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("training iteration")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(out, "compare.png"))
    plt.close(fig)


def fig_transfer(T_bt, T_ridge, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    labels = [f"{LAB[a]} it{i}" for a, i in CELLS]
    for ax, T, title in ((axes[0], T_bt, "(a) best-vs-worst direction"), (axes[1], T_ridge, "(b) all-8 direction")):
        sns.heatmap(T, ax=ax, cmap="RdBu_r", vmin=0.25, vmax=0.75, center=0.5, square=True, xticklabels=labels,
                    yticklabels=labels, cbar_kws={"label": "held-out: best projects above worst", "shrink": 0.75})
        ax.axhline(10, color="white", lw=2)
        ax.axvline(10, color="white", lw=2)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("tested on")
        ax.set_ylabel("fitted on")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "transfer.png"))
    plt.close(fig)


def fig_features(feat_df, out):
    names = list(FEATURES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, arm in zip(axes, ARMS):
        g = feat_df[feat_df.arm == arm].set_index("feature").loc[names]
        ax.axvline(0, color="#444444", lw=0.8)
        ax.scatter(g["reward"], y, marker="D", s=34, color="#222222", label="the reward itself", zorder=3)
        ax.scatter(g["bt"], y, marker="o", s=40, facecolors="white", edgecolors=COL[arm], linewidths=1.6,
                   label="best-vs-worst direction", zorder=3)
        ax.scatter(g["ridge"], y, marker="o", s=40, color=COL[arm], label="all-8 direction", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("within-round correlation with the feature")
        ax.set_title(f"{LAB[arm]}: what the directions and the reward favour")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=9)
    fig.savefig(os.path.join(out, "features.png"))
    plt.close(fig)


# ── features ──────────────────────────────────────────────────────────────────────────────────

def text_features(texts, lengths, degenerate) -> pd.DataFrame:
    cols = {}
    for name, rx in FEATURES.items():
        if name == "length (log chars)":
            cols[name] = np.log1p(lengths)
        elif name == "cut off mid-sentence":
            cols[name] = np.array([float(bool(t.strip()) and t.rstrip()[-1] not in ".!?\"')\u201d\u2019") for t in texts])
        elif name == "degenerate text":
            cols[name] = degenerate
        else:
            cols[name] = np.array([float(bool(rx.search(t))) for t in texts])
    return pd.DataFrame(cols)


def within_corr(x, y, cand_round, R):
    """Correlation of x and y after removing each round's mean from both."""
    def centre(v):
        cnt = np.bincount(cand_round, minlength=R)
        return v - (np.bincount(cand_round, weights=v, minlength=R) / np.maximum(cnt, 1))[cand_round]
    xc, yc = centre(x), centre(y)
    d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / d) if d > 0 else np.nan


def render(res, gq, T, feat, fig_dir):
    style()
    fig_analysis(res, gq[gq.method == "bt"], fig_dir, "bt", "meandiff", "bw",
                 "held-out: best projects above worst", "quintile of the round's reward gap (best − worst)",
                 ("Bradley-Terry direction", "mean of best − worst"), "analysis1_best_worst.png")
    fig_analysis(res, gq[gq.method == "ridge"], fig_dir, "ridge", "advmean", "rho",
                 "held-out: within-round Spearman\n(projection vs reward, all 8)",
                 "quintile of the round's reward spread (std)",
                 ("ridge direction", "advantage-weighted mean"), "analysis2_all8.png")
    fig_compare(res, fig_dir)
    fig_transfer(T["bt"], T["ridge"], fig_dir)
    fig_features(feat, fig_dir)


# ── main ──────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", default=DEFAULT_SHARE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--plots-only", action="store_true", help="re-render the figures from the saved tables")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    fig_dir, tab_dir = os.path.join(out, "figures"), os.path.join(out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    if a.plots_only:
        res = pd.read_csv(os.path.join(tab_dir, "scores_by_iteration.csv"))
        gq = pd.read_csv(os.path.join(tab_dir, "by_reward_difference.csv"))
        T = {m: pd.read_csv(os.path.join(tab_dir, f"transfer_{n}.csv"), index_col=0)
             for m, n in (("bt", "best_worst"), ("ridge", "all8"))}
        render(res, gq, T, pd.read_csv(os.path.join(tab_dir, "features.csv")), fig_dir)
        return

    print("loading ...", flush=True)
    D = Data(a.share, os.path.join(out, ".emb_cache"))
    A = Analysis(D)

    print("choosing the regularisation (mean held-out score over all 20 arm x iteration cells) ...", flush=True)
    C, C_scores = A.select("bt", C_GRID, "bw")
    alpha, alpha_scores = A.select("ridge", ALPHA_GRID, "rho")
    json.dump({"bradley_terry_C": C, "C_grid_mean_bw": C_scores, "ridge_alpha": alpha, "alpha_grid_mean_rho": alpha_scores},
              open(os.path.join(tab_dir, "hyperparameters.json"), "w"), indent=1)

    print("held-out scores ...", flush=True)
    rows, per_round, folds_w = [], [], {}
    for cell in CELLS:
        row = dict(arm=cell[0], iteration=cell[1], n_rounds=len(A.cell_rounds[cell]),
                   n_patients=int(D.rounds.patient.iloc[A.cell_rounds[cell]].nunique()),
                   share_clean_rounds=float(D.rounds.clean.iloc[A.cell_rounds[cell]].mean()))
        for method, hp in (("bt", C), ("meandiff", None), ("ridge", alpha), ("advmean", None)):
            sc, ws = A.cv(method, cell, hp)
            folds_w[(method, cell)] = ws
            for s in ("bw", "allpairs", "rho"):
                row[f"{s}_{method}"] = float(sc[s].mean())
                row[f"{s}_{method}_lo"], row[f"{s}_{method}_hi"] = cluster_ci(sc, s, rng)
                row[f"{s}_{method}_clean"] = float(sc.loc[sc.clean, s].mean())   # same direction, clean rounds only
            if method in ("bt", "ridge"):
                sc = sc.assign(method=method, arm=cell[0], iteration=cell[1])
                per_round.append(sc)
        sl = round_scores(D, D.length, A.cell_rounds[cell])
        for s in ("bw", "allpairs", "rho"):
            row[f"{s}_length"] = float(sl[s].mean())
        wb, wr = A.fit("bt", A.cell_rounds[cell], C), A.fit("ridge", A.cell_rounds[cell], alpha)
        row["cos_bt_ridge"] = float(wb @ wr)
        row["reliability_bt"] = A.split_half("bt", cell, C, rng)
        row["reliability_ridge"] = A.split_half("ridge", cell, alpha, rng)
        rows.append(row)
        print(f"  {cell[0]} it{cell[1]:2d}: bw bt {row['bw_bt']:.3f} ridge {row['bw_ridge']:.3f} len {row['bw_length']:.3f} | "
              f"rho bt {row['rho_bt']:.3f} ridge {row['rho_ridge']:.3f} | rel bt {row['reliability_bt']:.2f} "
              f"ridge {row['reliability_ridge']:.2f} | cos {row['cos_bt_ridge']:.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(tab_dir, "scores_by_iteration.csv"), index=False)
    pr = pd.concat(per_round)
    pr = pr.merge(D.rounds[["gap", "spread"]].reset_index().rename(columns={"index": "round_idx"}), on="round_idx")

    # accuracy by the round's reward gap (analysis 1) / reward spread (analysis 2), quintiles within arm
    gq = []
    for method, col in (("bt", "gap"), ("ridge", "spread")):
        for arm in ARMS:
            g = pr[(pr.method == method) & (pr.arm == arm)].copy()
            g["quintile"] = pd.qcut(g[col].rank(method="first"), 5, labels=range(1, 6)).astype(int)
            s = g.groupby("quintile").agg(bw=("bw", "mean"), allpairs=("allpairs", "mean"), rho=("rho", "mean"),
                                          lo=(col, "min"), hi=(col, "max"), n=("bw", "size")).reset_index()
            gq.append(s.assign(arm=arm, method=method, by=col))
    gq = pd.concat(gq)
    gq.round(4).to_csv(os.path.join(tab_dir, "by_reward_difference.csv"), index=False)

    # transfer: fit on one cell (training folds), test on every cell's held-out fold
    T = {}
    for method in ("bt", "ridge"):
        M = np.zeros((len(CELLS), len(CELLS)))
        for i, ci in enumerate(CELLS):
            for j, cj in enumerate(CELLS):
                hits = n = 0
                for f, w in folds_w[(method, ci)].items():
                    te = A.cell_rounds[cj][A.fold[A.cell_rounds[cj]] == f]
                    s = D.delta[te] @ w
                    hits += (s > 0).sum() + 0.5 * (s == 0).sum()
                    n += len(te)
                M[i, j] = hits / n
        names = [f"{a}_it{i}" for a, i in CELLS]
        T[method] = pd.DataFrame(M, index=names, columns=names)
        T[method].round(4).to_csv(os.path.join(tab_dir, f"transfer_{'best_worst' if method == 'bt' else 'all8'}.csv"))

    # what each arm's pooled direction (all iterations) and the reward itself favour
    F = text_features(D.text, D.length, D.degenerate)
    R = len(D.rounds)
    pooled, feat_rows = {}, []
    for arm in ARMS:
        rounds = np.flatnonzero((D.rounds.arm == arm).to_numpy())
        c = A.cands_of(rounds)
        pooled[arm] = {"bt": A.fit("bt", rounds, C), "ridge": A.fit("ridge", rounds, alpha)}
        proj = {m: D.E[c] @ w for m, w in pooled[arm].items()}
        sub_round = D.cand_round[c]
        for name in F.columns:
            x = F[name].to_numpy()[c]
            feat_rows.append(dict(arm=arm, feature=name, share_or_mean=float(x.mean()),
                                  reward=within_corr(x, D.adv[c], sub_round, R),
                                  bt=within_corr(x, proj["bt"], sub_round, R),
                                  ridge=within_corr(x, proj["ridge"], sub_round, R)))
    feat = pd.DataFrame(feat_rows)
    feat.round(4).to_csv(os.path.join(tab_dir, "features.csv"), index=False)

    # examples: per arm, among rounds without degenerate text, the 3 the all-8 direction spreads furthest apart
    lines = ["# Example rounds\n",
             "Per arm, among rounds with no degenerate candidate: the 3 rounds whose candidates the pooled all-8",
             "direction spreads furthest apart; the candidate it scores highest and lowest, with their rewards.",
             "Selected by the direction only, never by the reward.\n"]
    for arm in ARMS:
        rounds = np.flatnonzero((D.rounds.arm == arm).to_numpy() & D.rounds.clean.to_numpy())
        spread = []
        for k in rounds:
            s, e = D.rounds.start.iat[k], D.rounds.end.iat[k]
            p = D.E[s:e] @ pooled[arm]["ridge"]
            spread.append((p.max() - p.min(), k, s + int(p.argmax()), s + int(p.argmin())))
        lines.append(f"\n## {LAB[arm]}\n")
        for _, k, hi, lo in sorted(spread, reverse=True)[:3]:
            rr = D.rounds.iloc[k]
            lines.append(f"\n**Iteration {rr.iteration}, patient {rr.patient}, turn {rr.turn}** "
                         f"(round rewards {D.r[rr.start:rr.end].min():.2f}–{D.r[rr.start:rr.end].max():.2f})\n")
            for tag, ci in (("highest", hi), ("lowest", lo)):
                txt = D.text[ci].replace("\n", " ")
                lines.append(f"- {tag} on the direction, reward {D.r[ci]:.2f}: {txt[:500]}{'…' if len(txt) > 500 else ''}")
    open(os.path.join(tab_dir, "examples.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    render(res, gq, T, feat, fig_dir)
    print("done:", out)


if __name__ == "__main__":
    main()
