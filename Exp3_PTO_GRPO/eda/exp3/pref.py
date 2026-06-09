"""
pref.py — Exp3 successor to the Exp2 ``pref_emb`` analysis (PTO-only).

Over the PTO ``pref_pairs/pairs.csv`` across iterations:
- chosen−rejected **score-margin** distributions (how decisive the τ-filtered pairs are);
- **sentence-embedding geometry** of chosen vs rejected completions (cached to disk, same
  idea as Exp2's ``emb_cache_words``) — within-pair cosine separation + a 2D projection,
  and how it drifts across iterations;
- **lexical features** distinguishing chosen from rejected over time (length, questions,
  affirmation) — cross-links to :mod:`behavior` to test "is the policy increasingly
  preferring affirmation-heavy turns?".

GRPO has no preference data, so everything here is empty for GRPO arms.
"""

import hashlib
import os
import pickle
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from . import WORKSPACE_ROOT
from .training import load_pref_pairs  # re-exported convenience

_RE_AFFIRM = re.compile(r"\byou are\b|\byou're (worthy|enough|strong|powerful|brave|amazing|a )", re.I)
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".emb_cache")
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ── Lexical features ─────────────────────────────────────────────────────────
def add_text_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add chosen/rejected length, question count, affirmation flag + their deltas."""
    out = pairs.copy()
    for side in ("chosen", "rejected"):
        s = out[side].astype(str)
        out[f"{side}_len"] = s.str.len()
        out[f"{side}_q"] = s.str.count(r"\?")
        out[f"{side}_affirm"] = s.apply(lambda t: bool(_RE_AFFIRM.search(t)))
    out["len_delta"] = out["chosen_len"] - out["rejected_len"]
    out["q_delta"] = out["chosen_q"] - out["rejected_q"]
    out["affirm_delta"] = out["chosen_affirm"].astype(int) - out["rejected_affirm"].astype(int)
    return out


# ── Embeddings (cached) ──────────────────────────────────────────────────────
def _embed_texts(texts: List[str], model_name: str = _DEFAULT_MODEL,
                 cache_dir: str = _CACHE_DIR) -> dict:
    """Return ``{text: vector}`` for unique texts, caching embeddings to disk by sha1."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{model_name.replace('/', '_')}.pkl")
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = pickle.load(open(cache_path, "rb"))
        except Exception:
            cache = {}
    uniq = {t for t in texts if isinstance(t, str) and t.strip()}
    todo = [t for t in uniq if _key(t) not in cache]
    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vecs = model.encode(todo, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True)
        for t, v in zip(todo, vecs):
            cache[_key(t)] = np.asarray(v, dtype=np.float32)
        pickle.dump(cache, open(cache_path, "wb"))
    return {t: cache[_key(t)] for t in uniq if _key(t) in cache}


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_pairs(pairs: pd.DataFrame, model_name: str = _DEFAULT_MODEL) -> pd.DataFrame:
    """Attach ``chosen_emb`` / ``rejected_emb`` (np arrays) + within-pair cosine separation."""
    if pairs.empty:
        return pairs
    lut = _embed_texts(pairs["chosen"].tolist() + pairs["rejected"].tolist(), model_name)
    out = pairs.copy()
    out["chosen_emb"] = out["chosen"].map(lambda t: lut.get(t))
    out["rejected_emb"] = out["rejected"].map(lambda t: lut.get(t))
    def _cos(r):
        a, b = r["chosen_emb"], r["rejected_emb"]
        return float(np.dot(a, b)) if a is not None and b is not None else np.nan
    out["cos_sep"] = out.apply(_cos, axis=1)  # 1 = identical direction, lower = more separated
    return out


def separation_by_iter(embedded_pairs: pd.DataFrame) -> pd.DataFrame:
    """Per iteration: mean within-pair cosine + mean score margin + lexical deltas."""
    df = embedded_pairs
    cols = {"cos_sep": "mean", "margin": "mean"}
    for c in ("len_delta", "q_delta", "affirm_delta"):
        if c in df.columns:
            cols[c] = "mean"
    return (df.groupby(["arm", "train_iter"], observed=True).agg(cols)
            .reset_index().sort_values(["arm", "train_iter"]))


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_margin_trajectory(pairs: pd.DataFrame, arm: Optional[str] = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    d = pairs if arm is None else pairs[pairs["arm"] == arm]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(d, x="train_iter", y="margin", color="#c5b0d5", ax=ax)
    ax.set_title(f"Preference margin (chosen − rejected score) by iteration"
                 + (f" — {arm}" if arm else ""))
    ax.set_xlabel("training iteration"); ax.set_ylabel("score margin")
    fig.tight_layout()
    return fig


def plot_pref_feature_drift(sep_by_iter: pd.DataFrame, arm: Optional[str] = None):
    """How chosen-vs-rejected separation + lexical preference drifts across iterations."""
    import matplotlib.pyplot as plt
    d = sep_by_iter if arm is None else sep_by_iter[sep_by_iter["arm"] == arm]
    metrics = [c for c in ["cos_sep", "len_delta", "q_delta", "affirm_delta"] if c in d.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), squeeze=False)
    for ax, m in zip(axes.flat, metrics):
        for a, g in d.groupby("arm"):
            ax.plot(g["train_iter"], g[m], marker="o", label=a)
        ax.axhline(0, color="grey", lw=0.6, ls="--") if m != "cos_sep" else None
        ax.set_title(m); ax.set_xlabel("iteration")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("What the policy increasingly prefers (chosen − rejected)", y=1.04, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_embedding_projection(embedded_pairs: pd.DataFrame, arm: Optional[str] = None,
                              iters: Optional[List[int]] = None, sample: int = 300):
    """2D PCA of chosen (●) vs rejected (×) completion embeddings, faceted by iteration."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    d = embedded_pairs if arm is None else embedded_pairs[embedded_pairs["arm"] == arm]
    d = d.dropna(subset=["chosen_emb", "rejected_emb"])
    if d.empty:
        return None
    iters = iters or sorted(d["train_iter"].unique())
    # fit PCA on the pooled embedding cloud for a shared projection
    pool = np.vstack(list(d["chosen_emb"]) + list(d["rejected_emb"]))
    pca = PCA(n_components=2).fit(pool)
    fig, axes = plt.subplots(1, len(iters), figsize=(3.2 * len(iters), 3.2), squeeze=False)
    for ax, it in zip(axes.flat, iters):
        gi = d[d["train_iter"] == it]
        if len(gi) > sample:
            gi = gi.sample(sample, random_state=0)
        ch = pca.transform(np.vstack(list(gi["chosen_emb"])))
        rj = pca.transform(np.vstack(list(gi["rejected_emb"])))
        ax.scatter(rj[:, 0], rj[:, 1], s=8, marker="x", c="#d62728", alpha=0.5, label="rejected")
        ax.scatter(ch[:, 0], ch[:, 1], s=8, marker="o", c="#2ca02c", alpha=0.5, label="chosen")
        ax.set_title(f"iter {it}"); ax.set_xticks([]); ax.set_yticks([])
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Chosen vs rejected completion embeddings (2D PCA)", y=1.04, fontweight="bold")
    fig.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   Mass Mean Probe — latent-space preference direction (archive pref_emb style) ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Per iteration, the unit "preference direction" = normalized mean(chosen-rejected)
# completion embeddings (a Mass Mean Probe). Projecting words / MI-concept lists onto
# it reads out WHAT the policy prefers; tracking it across iterations shows the drift.

_TOK = re.compile(r"[a-z]{3,}")

# MI behavior word lists (archive's Change/Sustain/Therapist + Exp3 affirmation/reflection/question
# to test the affirmation-drift hypothesis). Alphabetic, lowercase, >=3 chars.
MI_CATEGORIES = {
    "Affirmation":   ["proud", "strong", "brave", "capable", "worthy", "amazing", "wonderful",
                      "appreciate", "admire", "courage", "resilient", "beautiful", "inspiring"],
    "Reflection":    ["sounds", "seems", "feel", "feeling", "hearing", "sense", "reflect",
                      "understand", "acknowledge"],
    "OpenQuestion":  ["what", "how", "why", "tell", "describe", "explore", "wondering",
                      "curious", "share"],
    "ChangeTalk":    ["ready", "willing", "able", "reason", "need", "want", "change", "commit",
                      "desire", "goal", "hope"],
    "SustainTalk":   ["difficult", "problem", "struggle", "stuck", "impossible", "afraid", "hard",
                      "overwhelmed", "worried"],
    "TherapistActions": ["listen", "understand", "reflect", "summarize", "explore", "support",
                         "validate", "affirm", "encourage"],
}


def preference_direction_by_iter(embedded_pairs: pd.DataFrame) -> dict:
    """``{train_iter: unit direction}`` = normalized mean(chosen_emb - rejected_emb) per iteration."""
    out = {}
    d = embedded_pairs.dropna(subset=["chosen_emb", "rejected_emb"])
    for it, g in d.groupby("train_iter"):
        deltas = np.vstack(list(g["chosen_emb"])) - np.vstack(list(g["rejected_emb"]))
        v = deltas.mean(axis=0)
        out[int(it)] = v / (np.linalg.norm(v) + 1e-12)
    return out


def probe_quality_by_iter(embedded_pairs: pd.DataFrame, directions: dict) -> pd.DataFrame:
    """Per iter: ``wins_correct`` (% pairs where chosen projects higher) + ``mean_gap`` + margin.

    A real preference direction has ``wins_correct`` well above 0.5.
    """
    rows = []
    d = embedded_pairs.dropna(subset=["chosen_emb", "rejected_emb"])
    for it, g in d.groupby("train_iter"):
        it = int(it)
        if it not in directions:
            continue
        dv = directions[it]
        pc = np.vstack(list(g["chosen_emb"])) @ dv
        pr = np.vstack(list(g["rejected_emb"])) @ dv
        rows.append({"train_iter": it, "n": len(g), "wins_correct": float((pc > pr).mean()),
                     "mean_gap": float((pc - pr).mean()),
                     "mean_margin": float(g["margin"].mean()) if "margin" in g else np.nan})
    return pd.DataFrame(rows).sort_values("train_iter")


def build_vocab(pairs: pd.DataFrame, top_n: int = 4000) -> list:
    """Corpus-derived vocabulary: the most frequent alphabetic (>=3-char) words in chosen+rejected."""
    from collections import Counter
    c = Counter()
    for col in ("chosen", "rejected"):
        if col in pairs:
            for t in pairs[col].astype(str):
                c.update(_TOK.findall(t.lower()))
    return [w for w, _ in c.most_common(top_n)]


def embed_vocab(words: list, model_name: str = _DEFAULT_MODEL):
    """Return ``(present_words, matrix)`` of cached embeddings (aligned rows)."""
    lut = _embed_texts(words, model_name)
    present = [w for w in words if w in lut]
    mat = np.vstack([lut[w] for w in present]) if present else np.zeros((0, 1), dtype=np.float32)
    return present, mat


def word_projection(words: list, word_mat: np.ndarray, directions: dict) -> pd.DataFrame:
    """DataFrame (index=word, columns=iter) of word·direction projections + a ``mean`` column.

    Sort by an iteration column (or ``mean``) to read the top chosen- vs rejected-aligned words.
    """
    cols = {it: word_mat @ dv for it, dv in sorted(directions.items())}
    df = pd.DataFrame(cols, index=words)
    df["mean"] = df.mean(axis=1)
    return df


def category_projection(directions: dict, categories: dict = None,
                        model_name: str = _DEFAULT_MODEL) -> pd.DataFrame:
    """Per (MI category, iter): mean projection of the category's words onto the preference direction.

    Embeds the category word lists directly (not reliant on corpus overlap). Long format:
    ``category, train_iter, score, n_words``.
    """
    categories = categories or MI_CATEGORIES
    allwords = sorted({w for ws in categories.values() for w in ws})
    present, mat = embed_vocab(allwords, model_name)
    idx = {w: i for i, w in enumerate(present)}
    rows = []
    for cat, ws in categories.items():
        ii = [idx[w] for w in ws if w in idx]
        if not ii:
            continue
        sub = mat[ii]
        for it, dv in sorted(directions.items()):
            rows.append({"category": cat, "train_iter": it,
                         "score": float((sub @ dv).mean()), "n_words": len(ii)})
    return pd.DataFrame(rows)
