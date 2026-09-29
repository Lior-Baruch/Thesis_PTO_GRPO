"""
analyze_grpo_group_directions_2026-09-29.py — the win-minus-lose embedding direction, per GRPO round.

Doron's request (2026-09-29): redo the Exp2 preference-direction analysis
(``Exp2_PTO/eda/pref_emb/preference_analysis.ipynb``) on GRPO, per round, using the full group of 8.
Reads ONLY the Doron share (``grpo_groups/`` + ``conversations/``), so he can rerun it on his copy.

Same embedding as Exp2: ``Alibaba-NLP/gte-base-en-v1.5``, mean pooling, L2-normalised. Under
transformers 5 its remote code leaves non-persistent buffers uninitialised; :func:`load_gte` rebuilds
them (checked: it reproduces Exp2's cached word embeddings at cosine 1.00000).

Per (arm, training iteration), over the ``train`` rounds with at least two distinct rewards:

1. **Direction.** Per round Δ = emb(best) − emb(worst); the direction is the normalised mean Δ (Exp2's
   mass-mean probe). Also the advantage-weighted direction (what GRPO's update weights) for comparison.
2. **Held-out probe.** 5 folds by patient: fit on 4/5 of the patients, score the other 1/5. Two scores
   per round: does best project above worst (Exp2's "wins correct"), and the Spearman correlation of
   projection with reward across all the round's candidates. 95% CIs resample patients.
3. **Length controls.** Replies grow ~5x over training, so beside every score: a length-only predictor
   (longer = better), and the direction with its linear length component removed.
4. **Per round.** The distribution of cos(round Δ, held-out direction).
5. **Stability.** Cosines between all 20 directions, against each direction's split-half reliability.
6. **Does the model move that way?** The shift of the model's own replies (therapist turns >= 13 in
   ``conversations/``) from iter_{N-1} to iter_N, against iteration N's direction.
7. **Meaning.** Exp2's 47k-word vocabulary and MI word lists projected on each arm's direction, and the
   candidates that project highest and lowest.

Usage (repo root, repo .venv; local GPU, ~1 GB VRAM):
    .venv/Scripts/python.exe meetings/build/analyze_grpo_group_directions_2026-09-29.py \
        [--share <Exp3_share_for_Doron>] [--out <dir>] [--wordfreq-dir <pip --target dir>]
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch                                              # before transformers / sklearn (local sm_120)
from transformers import AutoModel, AutoTokenizer
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
import seaborn as sns                                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SHARE = r"G:\My Drive\Thesis_PTO_GRPO\Exp3_share_for_Doron"
DEFAULT_OUT = os.path.join(REPO, "meetings", "2026-09-29_doron_grpo_groups", "analysis")
EXP2_WORDS = os.path.join(REPO, "Exp2_PTO", "eda", "pref_emb", "emb_cache_words",
                          "words_0ba65b02_Alibaba-NLP_gte-base-en-v1.5.npy")

MODEL = "Alibaba-NLP/gte-base-en-v1.5"
ARMS = ["GRPO_K0", "GRPO_K5"]
ITERS = list(range(1, 11))
COL = {"GRPO_K0": "#d55e00", "GRPO_K5": "#e69f00"}       # the paper's K=0 / K=5 colours
MARK = {"GRPO_K0": "o", "GRPO_K5": "s"}
LAB = {"GRPO_K0": "K=0", "GRPO_K5": "K=5"}
N_FOLDS, N_BOOT, N_SPLITS, SEED = 5, 1000, 50, 0
MIN_TURN = 13                                             # the rounds sit at turns 13-51
MI_CATEGORIES = {                                         # verbatim from the Exp2 notebook
    "Change Talk": ["ready", "willing", "able", "reason", "need", "want", "change", "commit", "desire"],
    "Sustain Talk": ["difficult", "problem", "struggle", "stuck", "impossible", "afraid"],
    "Therapist Actions": ["listen", "understand", "reflect", "summarize", "explore", "support", "validate"],
}


# ── embedding ─────────────────────────────────────────────────────────────────────────────────
def load_gte():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
    m.embeddings.register_buffer("position_ids", torch.arange(m.config.max_position_embeddings), persistent=False)
    m.embeddings._init_rope(m.config)
    return tok, m.cuda()


def embed(texts: list, cache_dir: str, max_batch: int = 128, token_budget: int = 8192) -> np.ndarray:
    """Unit-norm gte embeddings (float32), cached on disk by the md5 of the text list.

    VRAM: the remote code uses plain attention, so memory grows with batch x length^2. A fixed batch
    of 128 reached 11.3 GB reserved on the 12 GB local card at the long end (2026-09-29) - and an
    over-budget request there reboots the PC rather than raising. So batches are sized by a token
    budget, and the allocator is capped so an overrun raises OutOfMemoryError instead.
    """
    key = hashlib.md5("\x00".join(texts).encode("utf-8")).hexdigest()[:12]
    path = os.path.join(cache_dir, f"gte_{key}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    torch.cuda.set_per_process_memory_fraction(0.5)
    tok, m = load_gte()
    order = np.argsort([len(t) for t in texts])           # length-sorted batches waste less padding
    batches, i = [], 0
    while i < len(order):                                 # chars/3 over-estimates tokens: a safe size
        n = max(1, min(max_batch, token_budget // max(1, min(512, len(texts[order[min(i + max_batch, len(order)) - 1]]) // 3))))
        batches.append(order[i:i + n])
        i += n
    out = np.zeros((len(texts), 768), dtype=np.float16)
    done = 0
    for b, idx in enumerate(batches):
        done += len(idx)
        x = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h = m(**x).last_hidden_state.float()
        mask = x["attention_mask"].unsqueeze(-1).float()
        e = torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1)
        out[idx] = e.cpu().numpy().astype(np.float16)
        if b % 200 == 0:
            print(f"    embedded {done:>7,}/{len(texts):,}  vram {torch.cuda.memory_reserved() / 1e9:.1f}G", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, out)
    del m
    torch.cuda.empty_cache()
    return out.astype(np.float32)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ── data ──────────────────────────────────────────────────────────────────────────────────────
def load_rounds(share: str) -> pd.DataFrame:
    """One row per candidate of every train round with >= 2 distinct rewards."""
    rows, skipped = [], {}
    for arm in ARMS:
        for it in ITERS:
            n_ties = 0
            for line in gzip.open(os.path.join(share, "grpo_groups", arm, f"iter_{it:02d}.jsonl.gz"), "rt", encoding="utf-8"):
                r = json.loads(line)
                if r["split"] != "train":
                    continue
                cands = [(i, c) for i, c in enumerate(r["candidates"]) if c["reward"] is not None]
                if len({c["reward"] for _, c in cands}) < 2:
                    n_ties += 1
                    continue
                for i, c in cands:
                    rows.append((arm, it, r["round"], r["epoch"], r["patient_id"], r["turn"], i,
                                 c["text"], float(c["reward"]), len(c["text"])))
            skipped[(arm, it)] = n_ties
    df = pd.DataFrame(rows, columns=["arm", "iteration", "round", "epoch", "patient_id", "turn", "cand",
                                     "text", "reward", "length"])
    return df, skipped


def load_replies(share: str) -> pd.DataFrame:
    """The model's own therapist replies at turn >= MIN_TURN, iter_00..iter_10, both arms."""
    rows = []
    for arm in ARMS:
        for k in range(0, 11):
            for p in sorted(glob.glob(os.path.join(share, "conversations", arm, f"iter_{k:02d}", "patient_*.csv"))):
                c = pd.read_csv(p, usecols=["turn", "speaker", "text"])
                c = c[(c.speaker == "therapist") & (c.turn >= MIN_TURN) & c.text.notna()]
                pid = int(re.search(r"patient_(\d+)", p).group(1))
                rows += [(arm, k, pid, str(t)) for t in c.text]
    return pd.DataFrame(rows, columns=["arm", "model_iter", "patient_id", "text"])


# ── per-round arrays ──────────────────────────────────────────────────────────────────────────
def round_arrays(g: pd.DataFrame, E: np.ndarray):
    """For one (arm, iteration): list of per-round dicts with embeddings, rewards, lengths."""
    out = []
    for (rid,), d in g.groupby(["round"], sort=True):
        assert len(d) <= 8 and d["patient_id"].nunique() == 1 and d["turn"].nunique() == 1, \
            f"round {rid} mixes rounds - is `round` unique within the file?"
        e = E[d["emb"].to_numpy()]
        r = d["reward"].to_numpy()
        L = d["length"].to_numpy().astype(float)
        best, worst = int(np.argmax(r)), int(np.argmin(r))
        a = (r - r.mean()) / (r.std(ddof=1) + 1e-4)       # GRPO's group-relative advantage
        out.append(dict(round=rid, patient=int(d["patient_id"].iloc[0]), e=e, r=r, L=L, best=best, worst=worst,
                        delta=e[best] - e[worst], adv=(a[:, None] * e).sum(0) / len(r)))
    return out


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    return np.nan if sx == 0 or sy == 0 else float(np.corrcoef(rx, ry)[0, 1])


def length_direction(rounds) -> np.ndarray:
    """Linear length direction: normalised sum of (standardised length) x embedding, centred."""
    e = np.vstack([r["e"] for r in rounds])
    L = np.concatenate([r["L"] for r in rounds])
    z = (L - L.mean()) / (L.std() + 1e-9)
    return unit((z[:, None] * (e - e.mean(0))).sum(0))


def score_round(r, d):
    p = r["e"] @ d
    return dict(win=float(p[r["best"]] > p[r["worst"]]), rho=spearman(p, r["r"]),
                cos=float(unit(r["delta"]) @ d))


def evaluate(rounds, folds: dict) -> pd.DataFrame:
    """Held-out per-round scores for the Δ direction, the length-removed Δ, the advantage direction,
    and the length-only predictor."""
    recs = []
    for f in range(N_FOLDS):
        train = [r for r in rounds if folds[r["patient"]] != f]
        test = [r for r in rounds if folds[r["patient"]] == f]
        if not train or not test:
            continue
        D = unit(np.mean([r["delta"] for r in train], 0))
        A = unit(np.mean([r["adv"] for r in train], 0))
        Ld = length_direction(train)
        Dp = unit(D - (D @ Ld) * Ld)
        for r in test:
            s, sp, sa = score_round(r, D), score_round(r, Dp), score_round(r, A)
            Lb, Lw = r["L"][r["best"]], r["L"][r["worst"]]
            recs.append(dict(patient=r["patient"], win=s["win"], rho=s["rho"], cos=s["cos"],
                             win_perp=sp["win"], rho_perp=sp["rho"],
                             win_adv=sa["win"], rho_adv=sa["rho"],
                             win_len=1.0 if Lb > Lw else (0.5 if Lb == Lw else 0.0),
                             rho_len=spearman(r["L"], r["r"])))
    return pd.DataFrame(recs)


def cluster_ci(df: pd.DataFrame, col: str, rng) -> tuple:
    """95% CI of the mean of ``col``, resampling patients."""
    per = df.groupby("patient")[col].agg(["sum", "count"])
    s, n = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def split_half(rounds, rng) -> float:
    pats = np.array(sorted({r["patient"] for r in rounds}))
    vals = []
    for _ in range(N_SPLITS):
        half = set(rng.permutation(pats)[: len(pats) // 2])
        a = [r["delta"] for r in rounds if r["patient"] in half]
        b = [r["delta"] for r in rounds if r["patient"] not in half]
        vals.append(float(unit(np.mean(a, 0)) @ unit(np.mean(b, 0))))
    return float(np.mean(vals))


# ── figures ───────────────────────────────────────────────────────────────────────────────────
def style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})


def line(ax, x, y, arm, **kw):
    ax.plot(x, y, color=COL[arm], marker=MARK[arm], ms=4.5, lw=1.8, label=LAB[arm], **kw)


def fig_probe(m: pd.DataFrame, out: str):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    variants = [("", "direction (Exp2 probe)"), ("_perp", "direction, length removed"), ("_len", "length only (longer wins)")]
    for c, (suf, title) in enumerate(variants):
        for row, (stat, ylab) in enumerate([("win", "held-out: best projects above worst"),
                                            ("rho", "held-out: within-round Spearman\n(projection vs reward, all candidates)")]):
            ax = axes[row, c]
            for arm in ARMS:
                g = m[m.arm == arm]
                line(ax, g.iteration, g[f"{stat}{suf}"], arm)
                if stat == "win":
                    ax.fill_between(g.iteration, g[f"win{suf}_lo"], g[f"win{suf}_hi"], color=COL[arm], alpha=0.18, lw=0)
            ax.axhline(0.5 if stat == "win" else 0.0, color="#555555", lw=1, ls=":")
            ax.set_xticks(ITERS)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("training iteration")
            if c == 0:
                ax.set_ylabel(ylab)
    axes[0, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "probe.png"))
    plt.close(fig)


def fig_similarity(C: pd.DataFrame, m: pd.DataFrame, out: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    labels = [f"{LAB[arm]} it{it}" for arm in ARMS for it in ITERS]
    sns.heatmap(C.to_numpy(), ax=a, cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "cosine", "shrink": 0.8})
    a.axhline(10, color="white", lw=2)
    a.axvline(10, color="white", lw=2)
    a.tick_params(labelsize=7)
    a.set_title("(a) cosine between the 20 directions")
    for arm in ARMS:
        g = m[m.arm == arm]
        line(b, g.iteration, g.split_half, arm)
    k = [C.iloc[i, 10 + i] for i in range(10)]
    b.plot(ITERS, k, color="#333333", marker="D", ms=4, lw=1.8, label="K=0 vs K=5, same iteration")
    b.axhline(0, color="#555555", lw=1, ls=":")
    b.set_ylim(-0.2, 1.0)
    b.set_xticks(ITERS)
    b.set_xlabel("training iteration")
    b.set_ylabel("cosine")
    b.set_title("(b) split-half reliability vs K=0-K=5 agreement")
    for ln, arm in zip(b.get_lines()[:2], ARMS):
        ln.set_label(f"{LAB[arm]}: split-half reliability")
    b.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "direction_similarity.png"))
    plt.close(fig)


def fig_per_round(q: pd.DataFrame, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        g = q[q.arm == arm]
        ax.fill_between(g.iteration, g.q10, g.q90, color=COL[arm], alpha=0.15, lw=0, label="10-90%")
        ax.fill_between(g.iteration, g.q25, g.q75, color=COL[arm], alpha=0.35, lw=0, label="25-75%")
        ax.plot(g.iteration, g.q50, color=COL[arm], marker=MARK[arm], ms=4.5, lw=1.8, label="median")
        ax.axhline(0, color="#555555", lw=1, ls=":")
        ax.set_xticks(ITERS)
        ax.set_xlabel("training iteration")
        ax.set_title(f"{LAB[arm]}: each round's best-minus-worst vs the held-out direction")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].set_ylabel("cosine (one value per round)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_round_alignment.png"))
    plt.close(fig)


def fig_policy(s: pd.DataFrame, traj: pd.DataFrame, out: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for arm in ARMS:
        other = [x for x in ARMS if x != arm][0]
        g = s[s.arm == arm]
        line(a, g.iteration, g.cos_own, arm)
        a.get_lines()[-1].set_label(f"{LAB[arm]} shift vs {LAB[arm]} direction")
        a.plot(g.iteration, g.cos_other, color=COL[arm], marker=MARK[arm], ms=4, lw=1.2, ls="--", mfc="white",
               label=f"{LAB[arm]} shift vs {LAB[other]} direction")
    a.axhline(0, color="#555555", lw=1, ls=":")
    a.set_xticks(ITERS)
    a.set_xlabel("training iteration N")
    a.set_ylabel("cosine")
    a.set_title("(a) shift of the model's replies (iter N-1 to N)\nvs iteration N's direction")
    a.legend(frameon=False, fontsize=8, loc="best")
    for arm in ARMS:
        g = traj[traj.arm == arm]
        line(b, g.model_iter, g.proj, arm)
    b.axhline(0, color="#555555", lw=1, ls=":")
    b.set_xticks(range(0, 11))
    b.set_xlabel("model (iter_XX in conversations/)")
    b.set_ylabel("projection, relative to iter_00")
    b.set_title("(b) the model's replies along its own arm's pooled direction")
    b.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "policy_shift.png"))
    plt.close(fig)


def fig_words(wp: pd.DataFrame, out: str, k: int = 12):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, arm in zip(axes, ARMS):
        s = wp[arm] - wp[arm].mean()                      # relative to the average word
        top = pd.concat([s.nlargest(k), s.nsmallest(k).iloc[::-1]])
        ax.barh(range(len(top))[::-1], top.values, color=["#2a78d6" if v > 0 else "#e34948" for v in top.values], height=0.75)
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels(top.index, fontsize=8)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlabel("projection on the pooled direction, minus the average word's")
        ax.set_title(f"{LAB[arm]}: words toward best (blue) / worst (red)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "words.png"))
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", default=DEFAULT_SHARE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--wordfreq-dir", default=None, help="a pip --target dir holding wordfreq, if not in the venv")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    fig_dir, tab_dir = os.path.join(out, "figures"), os.path.join(out, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    cache = os.path.join(out, ".emb_cache")
    rng = np.random.default_rng(SEED)

    print("loading rounds + replies ...", flush=True)
    cands, ties = load_rounds(a.share)
    replies = load_replies(a.share)
    texts = sorted(set(cands.text) | set(replies.text))
    print(f"  {len(cands):,} candidates in {cands.groupby(['arm', 'iteration', 'round']).ngroups:,} rounds; "
          f"{len(replies):,} replies; {len(texts):,} unique texts", flush=True)
    E = embed(texts, cache)
    pos = {t: i for i, t in enumerate(texts)}
    cands["emb"] = cands.text.map(pos)
    replies["emb"] = replies.text.map(pos)

    pats = np.arange(96)
    folds = dict(zip(rng.permutation(pats), np.arange(96) % N_FOLDS))

    metrics, per_round_q, directions, rounds_by = [], [], {}, {}
    for arm in ARMS:
        for it in ITERS:
            rounds = round_arrays(cands[(cands.arm == arm) & (cands.iteration == it)], E)
            rounds_by[(arm, it)] = rounds
            D = unit(np.mean([r["delta"] for r in rounds], 0))
            A = unit(np.mean([r["adv"] for r in rounds], 0))
            Ld = length_direction(rounds)
            directions[(arm, it)] = D
            ev = evaluate(rounds, folds)
            row = dict(arm=arm, iteration=it, n_rounds=len(rounds), n_tied_rounds_dropped=ties[(arm, it)],
                       n_patients=len({r["patient"] for r in rounds}),
                       median_length=float(np.median(np.concatenate([r["L"] for r in rounds]))),
                       cos_delta_vs_advantage_direction=float(D @ A), cos_direction_vs_length=float(D @ Ld),
                       split_half=split_half(rounds, rng), per_round_cos_mean=float(ev["cos"].mean()))
            for suf in ("", "_perp", "_adv", "_len"):
                row[f"win{suf}"] = float(ev[f"win{suf}"].mean())
                row[f"win{suf}_lo"], row[f"win{suf}_hi"] = cluster_ci(ev, f"win{suf}", rng)
                row[f"rho{suf}"] = float(ev[f"rho{suf}"].mean())
            metrics.append(row)
            qs = np.percentile(ev["cos"], [10, 25, 50, 75, 90])
            per_round_q.append(dict(arm=arm, iteration=it, **{f"q{p}": v for p, v in zip((10, 25, 50, 75, 90), qs)},
                                    share_positive=float((ev["cos"] > 0).mean())))
            print(f"  {arm} it{it:2d}: rounds {len(rounds):5d}  win {row['win']:.3f}  win_perp {row['win_perp']:.3f}  "
                  f"win_len {row['win_len']:.3f}  rho {row['rho']:.3f}  split-half {row['split_half']:.3f}", flush=True)
    m = pd.DataFrame(metrics)
    q = pd.DataFrame(per_round_q)
    m.round(4).to_csv(os.path.join(tab_dir, "probe_by_iteration.csv"), index=False)
    q.round(4).to_csv(os.path.join(tab_dir, "per_round_alignment.csv"), index=False)

    keys = [(arm, it) for arm in ARMS for it in ITERS]
    C = pd.DataFrame([[float(directions[x] @ directions[y]) for y in keys] for x in keys],
                     index=[f"{arm}_it{it}" for arm, it in keys], columns=[f"{arm}_it{it}" for arm, it in keys])
    C.round(4).to_csv(os.path.join(tab_dir, "direction_cosines.csv"))

    # pooled direction per arm = normalised mean Δ over every round of every iteration
    pooled = {arm: unit(np.mean([r["delta"] for it in ITERS for r in rounds_by[(arm, it)]], 0)) for arm in ARMS}

    # the model's own replies: per-iteration centroid (mean over conversations of each conversation's mean)
    cent = {}
    for (arm, k), g in replies.groupby(["arm", "model_iter"]):
        per_conv = [E[d.emb.to_numpy()].mean(0) for _, d in g.groupby("patient_id")]
        cent[(arm, k)] = np.mean(per_conv, 0)
    shift_rows, traj_rows = [], []
    for arm in ARMS:
        other = [x for x in ARMS if x != arm][0]
        for it in ITERS:
            s = cent[(arm, it)] - cent[(arm, it - 1)]
            shift_rows.append(dict(arm=arm, iteration=it, cos_own=float(unit(s) @ directions[(arm, it)]),
                                   cos_other=float(unit(s) @ directions[(other, it)]), shift_norm=float(np.linalg.norm(s))))
        for k in range(0, 11):
            traj_rows.append(dict(arm=arm, model_iter=k, proj=float((cent[(arm, k)] - cent[(arm, 0)]) @ pooled[arm]),
                                  proj_other=float((cent[(arm, k)] - cent[(arm, 0)]) @ pooled[other])))
    shifts, traj = pd.DataFrame(shift_rows), pd.DataFrame(traj_rows)
    shifts.round(4).to_csv(os.path.join(tab_dir, "policy_shift.csv"), index=False)
    traj.round(4).to_csv(os.path.join(tab_dir, "policy_trajectory.csv"), index=False)

    # words: Exp2's vocabulary and its cached gte embeddings (identical model + pooling)
    wp = None
    if a.wordfreq_dir:
        sys.path.append(a.wordfreq_dir)                   # appended, so it cannot shadow the venv's packages
    try:
        from wordfreq import top_n_list                   # only to rebuild Exp2's vocabulary list
    except ImportError:
        top_n_list = None
        print("  wordfreq not importable: skipping the word projection")
    if top_n_list is not None and os.path.exists(EXP2_WORDS):
        words = sorted({w.lower() for w in top_n_list("en", 50_000) if re.match(r"^[a-z]{3,}$", w.lower())})
        assert hashlib.md5("".join(words).encode()).hexdigest()[:8] == "0ba65b02"
        W = np.load(EXP2_WORDS).astype(np.float32)
        wp = pd.DataFrame({arm: W @ pooled[arm] for arm in ARMS}, index=words)
        wp["K5_minus_K0"] = wp["GRPO_K5"] - wp["GRPO_K0"]
        wp.round(5).to_csv(os.path.join(tab_dir, "word_projection_pooled.csv"))
        widx = {w: i for i, w in enumerate(words)}
        mi_rows = []
        for cat, ws in MI_CATEGORIES.items():
            vecs = W[[widx[w] for w in ws if w in widx]]
            for arm in ARMS:
                for it in ITERS:
                    mi_rows.append(dict(category=cat, arm=arm, iteration=it, mean_projection=float((vecs @ directions[(arm, it)]).mean())))
                mi_rows.append(dict(category=cat, arm=arm, iteration="pooled", mean_projection=float((vecs @ pooled[arm]).mean())))
        pd.DataFrame(mi_rows).round(5).to_csv(os.path.join(tab_dir, "mi_categories.csv"), index=False)

    # example candidates: highest / lowest projection on the arm's pooled direction, all iterations
    ex = []
    for arm in ARMS:
        g = cands[cands.arm == arm].copy()
        g["proj"] = E[g.emb.to_numpy()] @ pooled[arm]
        g = g.drop_duplicates("text")
        for side, sub in (("top", g.nlargest(6, "proj")), ("bottom", g.nsmallest(6, "proj"))):
            for _, r in sub.iterrows():
                ex.append(dict(arm=arm, side=side, iteration=r.iteration, proj=round(r.proj, 4),
                               reward=round(r.reward, 3), length=r.length, text=r.text))
    pd.DataFrame(ex).to_csv(os.path.join(tab_dir, "examples_pooled.csv"), index=False, encoding="utf-8-sig")

    style()
    fig_probe(m, fig_dir)
    fig_similarity(C, m, fig_dir)
    fig_per_round(q, fig_dir)
    fig_policy(shifts, traj, fig_dir)
    if wp is not None:
        fig_words(wp, fig_dir)
    print("done:", out)


if __name__ == "__main__":
    main()
