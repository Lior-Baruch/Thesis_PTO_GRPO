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
# Embedding cache lives beside the parquet cache at the eda/ root (NOT inside the package source).
_EDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../eda
_CACHE_DIR = os.path.join(_EDA_DIR, ".emb_cache")
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


def pref_word_ranking(word_projection: pd.DataFrame, *, top_n: int = 15,
                      title: Optional[str] = None):
    """Horizontal bar of the ``top_n`` most chosen- (green) vs rejected- (red) aligned words.

    Takes the :func:`word_projection` frame (index = word, a ``mean`` column = pooled projection
    onto the chosen−rejected preference direction). Pools over iterations via ``mean``. Returns a
    ``fig`` (the notebook saves/shows it) — lives here (not ``plotting.py``) so all PTO-preference
    code stays in the one PTO-only module. Used by ``5_Preference``.
    """
    import matplotlib.pyplot as plt
    if word_projection.empty or "mean" not in word_projection.columns:
        return None
    top = (word_projection.sort_values("mean", ascending=False).head(top_n).index.tolist()
           + word_projection.sort_values("mean").head(top_n).index.tolist())
    s = word_projection.loc[top, "mean"].sort_values()
    fig, ax = plt.subplots(figsize=(7, max(5, 0.22 * len(s))))
    s.plot.barh(ax=ax, color=(s > 0).map({True: "#2ca02c", False: "#d62728"}))
    ax.set_title(title or "Words by preference projection (green=chosen, red=rejected)")
    ax.set_xlabel("projection onto chosen − rejected direction")
    ax.axvline(0, color="grey", lw=0.6)
    fig.tight_layout()
    return fig


def top_words_by_iter(word_projection: pd.DataFrame, *, k: int = 8) -> pd.DataFrame:
    """Per-iteration read-out: the top-``k`` chosen-aligned and rejected-aligned words each iter.

    Returns one row per training iteration with two string columns (``chosen_top`` / ``rejected_top``)
    — the literal "what the policy was being pushed toward vs away from at iteration N" story, so the
    drift heatmap can be read in words. Sorted by iteration.
    """
    if word_projection is None or word_projection.empty:
        return pd.DataFrame()
    iter_cols = sorted(c for c in word_projection.columns if c != "mean")
    rows = []
    for it in iter_cols:
        s = word_projection[it].sort_values(ascending=False)
        chosen = ", ".join(s.head(k).index)
        rejected = ", ".join(s.tail(k).index[::-1])
        rows.append({"train_iter": it, "chosen_top": chosen, "rejected_top": rejected})
    return pd.DataFrame(rows)


def pref_word_drift_heatmap(word_projection: pd.DataFrame, *, top_n: int = 12,
                            title: Optional[str] = None):
    """Per-iteration drift of the top preferred/rejected words — rows=word, cols=iteration.

    Complements the pooled :func:`pref_word_ranking` (which collapses iterations via ``mean``):
    here each of the ``top_n`` most chosen-aligned + ``top_n`` most rejected-aligned words is a
    row and every training iteration a column, colored by its projection onto the chosen−rejected
    direction (green=chosen, red=rejected, diverging at 0). Reads out drift like "affirmation
    words rise late while question/small-talk words fall". No new compute — the
    :func:`word_projection` frame already carries the per-iteration columns.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if word_projection is None or word_projection.empty or "mean" not in word_projection.columns:
        return None
    iter_cols = sorted(c for c in word_projection.columns if c != "mean")
    if not iter_cols:
        return None
    rows = (word_projection.sort_values("mean", ascending=False).head(top_n).index.tolist()
            + word_projection.sort_values("mean").head(top_n).index.tolist())
    # de-dup (a word can't be both halves unless top_n is huge) keeping chosen→rejected order
    seen, ordered = set(), []
    for w in rows:
        if w not in seen:
            seen.add(w); ordered.append(w)
    sub = word_projection.loc[ordered, iter_cols]
    vmax = float(np.nanmax(np.abs(sub.values))) or 1.0
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(iter_cols) + 2), max(5.0, 0.3 * len(sub))))
    sns.heatmap(sub, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "projection onto chosen − rejected"}, ax=ax)
    ax.set_title(title or "Preferred-word drift across iterations (green=chosen, red=rejected)")
    ax.set_xlabel("training iteration"); ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_category_drift(category_long: pd.DataFrame, *, palette=None):
    """MI-concept preference drift: each :data:`MI_CATEGORIES` group's projection across iterations.

    Takes the :func:`category_projection` long frame (``category, train_iter, score``) and draws
    one line per MI concept — the direct visual test of the affirmation-rising / question-falling
    hypothesis (Affirmation climbing above 0 over training, OpenQuestion / SustainTalk falling).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if category_long is None or category_long.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(category_long, x="train_iter", y="score", hue="category", marker="o",
                 palette=palette, ax=ax)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title("MI-concept preference drift (projection onto chosen − rejected)")
    ax.set_xlabel("training iteration"); ax.set_ylabel("mean projection")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1.0), title="MI concept", frameon=False)
    fig.tight_layout()
    return fig


def preference_direction_drift(directions: dict) -> pd.DataFrame:
    """How the preference DIRECTION itself moves across iterations.

    Takes ``{iter: unit direction}`` (:func:`preference_direction_by_iter`). Returns per iteration:
    a 2D PCA embedding of the direction vectors (``pc1``/``pc2`` — so the drift is a path you can
    plot) + ``cos_prev`` (cosine similarity to the previous iteration's direction; ~1 = stable,
    lower = the policy's preference is re-orienting).
    """
    if not directions:
        return pd.DataFrame(columns=["train_iter", "pc1", "pc2", "cos_prev"])
    its = sorted(directions)
    M = np.vstack([directions[i] for i in its])
    if len(its) >= 2:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2).fit_transform(M)
    else:
        xy = np.zeros((1, 2))
    rows = []
    for n, it in enumerate(its):
        cos_prev = float(M[n] @ M[n - 1]) if n > 0 else np.nan
        rows.append({"train_iter": int(it), "pc1": float(xy[n, 0]), "pc2": float(xy[n, 1]),
                     "cos_prev": cos_prev})
    return pd.DataFrame(rows)


def plot_direction_drift(drift_df: pd.DataFrame, *, title: Optional[str] = None):
    """Path of the per-iteration preference direction in 2D PCA + the consecutive-cosine line."""
    import matplotlib.pyplot as plt
    if drift_df is None or drift_df.empty:
        return None
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    its = drift_df["train_iter"].to_numpy()
    axp.plot(drift_df["pc1"], drift_df["pc2"], "-", color="#999999", lw=1, zorder=1)
    sc = axp.scatter(drift_df["pc1"], drift_df["pc2"], c=its, cmap="viridis", s=70, zorder=2)
    for _, r in drift_df.iterrows():
        axp.annotate(int(r["train_iter"]), (r["pc1"], r["pc2"]), fontsize=7, va="bottom")
    axp.set_title("Preference direction drift (2D PCA; arrow = iterations)")
    axp.set_xlabel("PC1"); axp.set_ylabel("PC2")
    fig.colorbar(sc, ax=axp, label="iteration", fraction=0.046)
    axc.plot(drift_df["train_iter"], drift_df["cos_prev"], marker="o", color="#7b4fb0")
    axc.set_ylim(0, 1.02); axc.axhline(1.0, color="grey", lw=0.6, ls="--")
    axc.set_title("Stability: cos(dir_t, dir_{t-1})"); axc.set_xlabel("iteration")
    axc.set_ylabel("cosine to previous")
    fig.suptitle(title or "What the policy prefers — how the direction moves", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def learn_unlearn_words(word_projection: pd.DataFrame, *, k: int = 10) -> pd.DataFrame:
    """Per consecutive-iteration transition, the words whose preference rose/fell the most.

    Δ = projection(it+1) − projection(it) for each word over each transition; keeps the top-``k``
    gainers (newly preferred = "learned") and top-``k`` losers ("unlearned"). Long frame:
    ``from_iter, to_iter, transition, word, delta, direction``.
    """
    if word_projection is None or word_projection.empty:
        return pd.DataFrame()
    iters = sorted(c for c in word_projection.columns if c != "mean")
    rows = []
    for a, b in zip(iters[:-1], iters[1:]):
        delta = (word_projection[b] - word_projection[a]).sort_values()
        picks = list(delta.head(k).items()) + list(delta.tail(k).items())
        for w, d in picks:
            rows.append({"from_iter": a, "to_iter": b, "transition": f"{a}→{b}",
                         "word": w, "delta": float(d),
                         "direction": "learned" if d > 0 else "unlearned"})
    return pd.DataFrame(rows)


def plot_learn_unlearn(luw_df: pd.DataFrame, *, transitions: Optional[List] = None,
                       max_panels: int = 4, k: int = 10):
    """Small-multiples of the biggest 'learned' (green) vs 'unlearned' (red) words per transition."""
    import matplotlib.pyplot as plt
    if luw_df is None or luw_df.empty:
        return None
    allt = list(dict.fromkeys(luw_df["transition"]))
    if transitions is None:
        # evenly sample up to max_panels transitions across training
        if len(allt) > max_panels:
            idx = np.linspace(0, len(allt) - 1, max_panels).round().astype(int)
            transitions = [allt[i] for i in idx]
        else:
            transitions = allt
    fig, axes = plt.subplots(1, len(transitions), figsize=(3.6 * len(transitions), 4.4), squeeze=False)
    for ax, t in zip(axes.flat, transitions):
        d = luw_df[luw_df["transition"] == t].sort_values("delta")
        d = pd.concat([d.head(k), d.tail(k)]).drop_duplicates("word").sort_values("delta")
        ax.barh(d["word"], d["delta"], color=(d["delta"] > 0).map({True: "#2ca02c", False: "#d62728"}))
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(f"iter {t}", fontsize=9); ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Learned (green, +) vs unlearned (red, −) words across iterations",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


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
