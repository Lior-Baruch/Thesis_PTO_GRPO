"""
pref.py — what the training signal pushes the policy toward, for BOTH methods.

**Part 1 (original, PTO-only)** — over the PTO ``pref_pairs/pairs.csv`` across iterations:
- chosen−rejected **score-margin** distributions (how decisive the τ-filtered pairs are);
- **sentence-embedding geometry** of chosen vs rejected completions (cached to disk, same
  idea as Exp2's ``emb_cache_words``) — within-pair cosine separation + a 2D projection,
  and how it drifts across iterations;
- **lexical features** distinguishing chosen from rejected over time (length, questions,
  affirmation) — cross-links to :mod:`behavior` to test "is the policy increasingly
  preferring affirmation-heavy turns?".

**Part 2 (2026-08-02) — the update-weighted view, which both methods have.** GRPO has no
preference pairs, but "preference" was never the essential thing: both methods weight the
candidates of a group and push the policy along the weighted sum. Write the update direction as
``normalize(Σ_g w_g · emb(t_g))`` and the two methods differ only in the weights ``w``:

===========  ======================================================================
DPO (PTO)    ``+1`` on the recorded ``chosen``, ``−1`` on ``rejected``, 0 on the rest
             (the roles are logged, so this is the literal τ-filtered training pair)
GRPO         the standardized group-relative advantage ``(r_g − mean_g) / std_g``,
             i.e. what actually scales each completion's gradient
===========  ======================================================================

Weights are then rescaled per group to ``Σ|w| = 2`` — DPO's natural ±1 scale — so a "unit of
push" means the same thing on both sides and the probes are comparable. Every downstream probe
(word ranking, MI-concept projection, drift) already takes a plain ``{iter: direction}`` dict, so
it works unchanged for GRPO.

**Part 3 (2026-08-02) — does the training signal predict the eval move?** ``link_to_outcomes``
joins each iteration's preference features to the persona-paired eval delta that iteration's
update produced (``model_iter_n`` vs ``model_iter_{n-1}``), turning "what the update prefers" from
a description into a testable mechanism for the sycophancy story.
"""

import hashlib
import os
import pickle
import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import WORKSPACE_ROOT, RE_AFFIRM, RE_EFFUSIVE, BOOT_SEED
from .training import load_generations, load_pref_pairs  # re-exported convenience

_RE_AFFIRM = RE_AFFIRM   # shared lexical affirmation cue (see constants.py)
_RE_EFFUSIVE = RE_EFFUSIVE
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
    code stays in the one PTO-only module. Used by ``arms/preference``.
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
                 palette=palette, seed=BOOT_SEED, ax=ax)
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   ONE PROBE, BOTH METHODS — the update-weighted candidate view                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# See the module docstring, Part 2. Everything below reads generations.jsonl (which BOTH
# trainers write, one row per group with all candidates nested) rather than PTO's pairs.csv.

# The group = the set of completions that compete in ONE update. `conversation_id` is in the key
# because PTO's `branch_id` is trunk DEPTH, not a unique id (it repeats across conversations) —
# keying on branch_id alone silently pools unrelated conversations.
_GROUP_KEYS = ["arm", "train_iter", "conversation_id", "branch_id", "epoch"]


def _cell_rng(seed: int, arm: str, it) -> "np.random.Generator":
    """A generator seeded by (seed, arm, iteration) — NOT by position in a shared stream.

    Every sampling/splitting decision below is per (arm, iteration), so it must depend only on that
    cell. With one shared stream the draw for ``PTO_LA0`` iter 3 would change whenever another arm
    is added to the frame, and the same figure would differ between a K-filtered and an all-arms render
    on identical data.
    """
    import zlib
    # `it` may be -1 (the pooled-over-iterations block); the seed sequence needs non-negatives.
    return np.random.default_rng([int(seed), zlib.crc32(str(arm).encode()), int(it) + 1])


def load_weighted_candidates(arms: Optional[List] = None, *,
                             drop_zero_weight: bool = True) -> pd.DataFrame:
    """Every training candidate with the weight its method's update actually gives it.

    One row per candidate of every group that contributed a gradient, with ``weight`` set per
    method (module docstring, Part 2): DPO's recorded ±1 chosen/rejected roles for PTO, the
    standardized group-relative advantage for GRPO. Weights are rescaled within each group to
    ``Σ|w| = 2`` so the two methods' "one unit of push" is the same size — without that, GRPO's
    weights carry the group's reward spread and every cross-method contrast is a scale artifact.

    Also attaches the lexical features both methods are tested on (``n_questions``,
    ``affirm_marker``, ``overpraise_marker``; ``len_chars`` comes from the loader) and
    ``group_size``.

    Drops: the ``eval`` phase (TRL's eval-loop generations produce no gradient), candidates with no
    score, and every group that lacks an up- **and** a down-weighted side — for PTO a branch point
    where the τ filter emitted no pair (logged ``chosen``, no ``rejected``), for GRPO a group with
    zero reward spread. Neither produced a gradient. With ``drop_zero_weight`` (default) the
    surviving zero-weight rows go too: they contribute nothing to any weighted quantity, and for
    PTO that is 6 of every 8 candidates.
    """
    gens = load_generations(arms)
    if gens.empty:
        return pd.DataFrame()
    d = gens[(gens["phase"] != "eval") & gens["score"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    # epoch is NaN for PTO and a float progress marker for GRPO; groupby DROPS NaN keys, so pin it.
    d["epoch"] = d["epoch"].fillna(-1.0)

    d["weight"] = 0.0
    is_pto = d["method"] == "PTO"
    if is_pto.any():
        d.loc[is_pto, "weight"] = (d.loc[is_pto, "role"]
                                   .map({"chosen": 1.0, "rejected": -1.0}).fillna(0.0))
    is_grpo = d["method"] == "GRPO"
    if is_grpo.any():
        g = d[is_grpo].groupby(_GROUP_KEYS)["score"]
        mean = d.loc[is_grpo, "group_mean"].fillna(g.transform("mean"))
        std = d.loc[is_grpo, "group_std"].fillna(g.transform("std")).replace(0.0, np.nan)
        d.loc[is_grpo, "weight"] = ((d.loc[is_grpo, "score"] - mean) / std).fillna(0.0)

    d["group_size"] = d.groupby(_GROUP_KEYS)["score"].transform("size")
    # A group only pushes if it has BOTH an up- and a down-weighted side. For PTO that drops the
    # ~16% of branch points that logged a `chosen` but no `rejected` — the τ filter found no pair
    # there, so DPO never saw the group; keeping it would rescale that lone `chosen` to a full
    # +2 one-sided push the trainer never applied. GRPO's standardized advantages always straddle 0.
    grp = d.groupby(_GROUP_KEYS)["weight"]
    d = d[(grp.transform("max") > 0) & (grp.transform("min") < 0)].copy()
    if d.empty:
        return d
    d = _rescale_weights(d)

    comp = d["completion"].astype(str)
    d["n_questions"] = comp.str.count(r"\?")
    d["affirm_marker"] = comp.map(lambda t: bool(_RE_AFFIRM.search(t))).astype(float)
    d["overpraise_marker"] = comp.map(lambda t: bool(_RE_EFFUSIVE.search(t))).astype(float)
    if drop_zero_weight:
        d = d[d["weight"] != 0.0].copy()
    return d.reset_index(drop=True)


def _rescale_weights(d: pd.DataFrame) -> pd.DataFrame:
    """Rescale within each group to ``Σ|w| = 2`` — DPO's ±1 pair size, the shared unit of push."""
    out = d.assign(_absw=d["weight"].abs())
    absum = out.groupby(_GROUP_KEYS)["_absw"].transform("sum")
    out = out[absum > 0].copy()
    out["weight"] = 2.0 * out["weight"] / out.groupby(_GROUP_KEYS)["_absw"].transform("sum")
    return out.drop(columns=["_absw"])


def reweight(cands: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Re-weight the SAME groups under the OTHER method's rule — the counterfactual.

    The cross-method direction cosine confounds two things: PTO and GRPO see different candidate
    **pools** (one branches a trunk, the other samples a group off a fixed prompt) *and* apply
    different weighting **rules**. Holding the groups fixed and swapping only the rule separates
    them, and every group already carries all its candidates' scores, so this needs no new data.

    ``rule``:

    - ``"dpo"`` — ``+1`` on the group's highest-scoring candidate, ``−1`` on its lowest, 0 for the
      rest (a best-vs-worst preference pair). On PTO's own groups this reconstructs what DPO did,
      which is the check that the reconstruction is faithful.
    - ``"grpo"`` — the standardized advantage ``(score − mean) / std`` over the group, computed
      from the candidates themselves (population std, matching the trainer).

    Pass a frame from ``load_weighted_candidates(..., drop_zero_weight=False)``: the counterfactual
    needs every candidate of the group, not just the two DPO kept. Weights are rescaled to the same
    ``Σ|w| = 2``, so a direction built from the result is directly comparable to the native one.
    """
    if cands.empty:
        return cands
    if rule not in ("dpo", "grpo"):
        raise ValueError(f"unknown rule {rule!r} (expected 'dpo' or 'grpo')")
    d = cands.copy()
    g = d.groupby(_GROUP_KEYS)["score"]
    if rule == "grpo":
        mean, std = g.transform("mean"), g.transform(lambda s: s.std(ddof=0))
        d["weight"] = ((d["score"] - mean) / std.replace(0.0, np.nan)).fillna(0.0)
    else:
        hi, lo = g.transform("max"), g.transform("min")
        # A tie for the extreme would double-count; keep the first occurrence only.
        first_hi = (d["score"] == hi) & ~d.duplicated(subset=_GROUP_KEYS + ["score"]) & (hi != lo)
        first_lo = (d["score"] == lo) & ~d.duplicated(subset=_GROUP_KEYS + ["score"]) & (hi != lo)
        d["weight"] = first_hi.astype(float) - first_lo.astype(float)
    return _rescale_weights(d).reset_index(drop=True)


def sample_groups(cands: pd.DataFrame, *, max_groups_per_iter: int = 400,
                  seed: int = BOOT_SEED, verbose: bool = True) -> pd.DataFrame:
    """Cap the number of GROUPS per (arm, iteration) — the embedding budget knob.

    Sampling is at the GROUP level so a group is never half-embedded, and seeded so a re-render
    reproduces the same figures. The default 400 was chosen by measuring, not guessed: split-half
    reliability of the per-iteration direction climbs 0.19 → 0.26 → 0.47 → 0.66 (GRPO) across caps
    50/100/200/400, so 400 is where the per-iteration estimate becomes usable for that method —
    while embedding all ~223k candidates would cost ~5x more for the next increment. PTO's
    per-iteration direction reaches only ~0.19 at this cap (PTO_LA5 ~0.30) — and unlike the GRPO
    figures above that is NOT an uncapped ceiling. ⚠ PTO produces 281–832 pairs an iteration
    (median 479 for LA0, 610 for LA5), so the cap BITES on 16 of 20 PTO arm-iterations (LA0 6/10,
    LA5 10/10) and discards 3,530 of 11,351 PTO groups (31%). Its per-iteration readouts therefore
    carry a reliability caveat that is partly self-inflicted, and its cross-arm claims use
    :func:`direction_by_arm`; raise the cap before concluding anything about PTO's per-iteration
    direction.

    Prints what it dropped unless ``verbose=False``: a cap that silently shrinks the evidence reads
    as full coverage in the artifact.
    """
    if cands.empty or not max_groups_per_iter:
        return cands
    keep, dropped, total = [], 0, 0
    for (arm, it), g in cands.groupby(["arm", "train_iter"], sort=True):
        ids = g.groupby(_GROUP_KEYS).ngroup()
        uniq = np.array(sorted(ids.unique()))
        total += len(uniq)
        if len(uniq) <= max_groups_per_iter:
            keep.append(g)
            continue
        chosen = set(_cell_rng(seed, arm, it)
                     .choice(uniq, size=max_groups_per_iter, replace=False).tolist())
        keep.append(g[ids.isin(chosen)])
        dropped += len(uniq) - max_groups_per_iter
    out = pd.concat(keep, ignore_index=True) if keep else cands.iloc[:0]
    if verbose:
        print(f"[sample_groups] cap={max_groups_per_iter}/iter, seed={seed}: kept "
              f"{total - dropped}/{total} groups ({dropped} dropped), {len(out)} candidate rows.")
    return out


def embed_candidates(cands: pd.DataFrame, model_name: str = _DEFAULT_MODEL) -> pd.DataFrame:
    """Attach ``emb`` to each weighted candidate (same disk cache as :func:`embed_pairs`)."""
    if cands.empty:
        return cands
    lut = _embed_texts(cands["completion"].astype(str).tolist(), model_name)
    out = cands.copy()
    out["emb"] = out["completion"].astype(str).map(lambda t: lut.get(t))
    return out[out["emb"].notna()].copy()


def _direction(sub: pd.DataFrame) -> np.ndarray:
    v = (np.vstack(list(sub["emb"])) * sub["weight"].to_numpy()[:, None]).sum(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def direction_by_iter(embedded: pd.DataFrame) -> dict:
    """``{arm: {train_iter: unit direction}}`` — the update direction, per arm per iteration.

    The generalization of :func:`preference_direction_by_iter` to any weighting (and therefore to
    GRPO): ``normalize(Σ w · emb)``. With DPO's ±1 weights the two agree by construction, which is
    what :func:`direction_agreement_with_pairs` checks on real data.
    """
    out: dict = {}
    for (arm, it), g in embedded.groupby(["arm", "train_iter"]):
        out.setdefault(arm, {})[int(it)] = _direction(g)
    return out


def _wins(sub: pd.DataFrame, dv: np.ndarray):
    """(share of groups ordered correctly by *dv*, mean projection gap) for one set of groups."""
    proj = np.vstack(list(sub["emb"])) @ dv
    gg = sub.assign(_proj=proj, _gid=sub.groupby(_GROUP_KEYS).ngroup())
    wins, gaps = [], []
    for _, grp in gg.groupby("_gid"):
        hi, lo = grp.loc[grp["weight"].idxmax()], grp.loc[grp["weight"].idxmin()]
        wins.append(hi["_proj"] > lo["_proj"])
        gaps.append(hi["_proj"] - lo["_proj"])
    return (float(np.mean(wins)) if wins else np.nan,
            float(np.mean(gaps)) if gaps else np.nan)


def _quality_row(sub: pd.DataFrame, dv: np.ndarray, rng) -> dict:
    """Probe quality for one (already grouped) block: in-sample wins, held-out wins, split-half."""
    gids = sub.groupby(_GROUP_KEYS).ngroup()
    uniq = np.array(sorted(gids.unique()))
    wins_in, gap = _wins(sub, dv)
    row = {"n_groups": len(uniq), "n_candidates": len(sub), "wins_correct": wins_in,
           "mean_gap": gap, "wins_holdout": np.nan, "split_half_cos": np.nan}
    if len(uniq) < 4:
        return row
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False).tolist())
    a, b = sub[gids.isin(half)], sub[~gids.isin(half)]
    if not len(a) or not len(b):
        return row
    da, db = _direction(a), _direction(b)
    row["split_half_cos"] = float(da @ db)
    # Each half is scored by the OTHER half's direction — no candidate helps fit the direction it
    # is then judged by, which is what makes this number honest.
    row["wins_holdout"] = float(np.mean([_wins(b, da)[0], _wins(a, db)[0]]))
    return row


def direction_quality(embedded: pd.DataFrame, directions: dict, *,
                      seed: int = BOOT_SEED) -> pd.DataFrame:
    """Per (arm, iter): is the direction real, and is it estimated from enough groups?

    - ``wins_correct`` — share of groups whose most up-weighted candidate projects above its most
      down-weighted one. **In-sample**, so it is optimistic: the direction was fitted on these very
      groups, and the smaller the iteration the more it overfits (measured: PTO's drops 0.88 → 0.68
      as the group count goes 50 → 380).
    - ``wins_holdout`` — the same share with each half judged by the *other* half's direction. This
      is the number to quote.
    - ``mean_gap`` — mean projection gap between the two extremes, in units of the unit direction.
    - ``split_half_cos`` — cosine between directions estimated on two disjoint halves. The
      **precision** check the original probe lacked, and it bites: a per-iteration PTO direction
      scores ~0.1–0.2 here, i.e. two halves of the same iteration point almost independently. Low
      ``wins_holdout`` means the axis is weak; low ``split_half_cos`` means it is real but not yet
      measured, and the fix is more groups (or :func:`direction_by_arm`, which pools them).
    """
    rows = []
    for (arm, it), g in embedded.groupby(["arm", "train_iter"]):
        dv = directions.get(arm, {}).get(int(it))
        if dv is None:
            continue
        rows.append({"arm": arm, "method": g["method"].iloc[0], "train_iter": int(it),
                     **_quality_row(g, dv, _cell_rng(seed, arm, it))})
    return pd.DataFrame(rows).sort_values(["arm", "train_iter"]).reset_index(drop=True)


def direction_by_arm(embedded: pd.DataFrame) -> dict:
    """``{arm: unit direction}`` pooled over every iteration — the arm's overall update target.

    The answer to :func:`direction_quality`'s reliability problem when the question is "what does
    this method prefer?" rather than "how does that move per iteration": pooling multiplies the
    group count by the iteration count, and the split-half cosine rises accordingly (see
    :func:`pooled_direction_quality`). Use the pooled direction for cross-method claims and the
    per-iteration ones only for drift, with their reliability quoted alongside.
    """
    return {arm: _direction(g) for arm, g in embedded.groupby("arm")}


def pooled_direction_quality(embedded: pd.DataFrame, directions_by_arm: dict, *,
                             seed: int = BOOT_SEED) -> pd.DataFrame:
    """:func:`direction_quality` for the pooled per-arm directions (one row per arm)."""
    rows = []
    for arm, g in embedded.groupby("arm"):
        dv = directions_by_arm.get(arm)
        if dv is None:
            continue
        rows.append({"arm": arm, "method": g["method"].iloc[0], "n_iters": g["train_iter"].nunique(),
                     **_quality_row(g, dv, _cell_rng(seed, arm, -1))})
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def _spearman_brown(split_half: float) -> float:
    """Half-sample split-half cosine -> reliability of the FULL-sample direction."""
    if split_half is None or not np.isfinite(split_half) or split_half <= 0:
        return np.nan
    return 2.0 * split_half / (1.0 + split_half)


def direction_cosine(directions: dict, arm_a: str, arm_b: str,
                     quality: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per matched iteration: cosine between two arms' update directions (1 = same target).

    Serves both cross-arm questions with one function — PTO vs GRPO at matched K ("do the two
    losses pull toward the same language?") and K=0 vs K=5 within a method ("does look-ahead
    re-aim the update?").

    Pass ``quality`` (a :func:`direction_quality` frame) to also get the **attenuation ceiling**:
    two noisily-estimated directions cannot correlate to 1 even if they are identical, so a raw
    cosine of 0.3 means different things at reliability 0.9 and at reliability 0.2. ``ceiling`` =
    ``sqrt(r_a · r_b)`` with each ``r`` the Spearman-Brown-corrected split-half reliability, and
    ``cosine_corrected`` = ``cosine / ceiling`` — the same correction
    :mod:`~eda_analysis.reliability` applies to cross-judge agreement, for the same reason.
    """
    da, db = directions.get(arm_a, {}), directions.get(arm_b, {})
    rel = {}
    if quality is not None and not quality.empty:
        rel = {(r.arm, int(r.train_iter)): _spearman_brown(r.split_half_cos)
               for r in quality.itertuples(index=False)}
    rows = []
    for i in sorted(set(da) & set(db)):
        row = {"train_iter": i, "arm_a": arm_a, "arm_b": arm_b, "cosine": float(da[i] @ db[i])}
        ra, rb = rel.get((arm_a, i), np.nan), rel.get((arm_b, i), np.nan)
        ceil = float(np.sqrt(ra * rb)) if np.isfinite(ra) and np.isfinite(rb) else np.nan
        row["ceiling"] = ceil
        row["cosine_corrected"] = row["cosine"] / ceil if ceil and np.isfinite(ceil) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_direction_cosines(directions_by_arm: dict,
                             quality: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Every arm pair's pooled direction cosine + attenuation ceiling (one row per pair).

    The cross-method and cross-K read that the per-iteration cosines are too noisy to support —
    pooling raises each direction's reliability enough for the comparison to mean something.
    """
    rel = {}
    if quality is not None and not quality.empty:
        rel = {r.arm: _spearman_brown(r.split_half_cos) for r in quality.itertuples(index=False)}
    arms = sorted(directions_by_arm)
    rows = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            cos = float(directions_by_arm[a] @ directions_by_arm[b])
            ra, rb = rel.get(a, np.nan), rel.get(b, np.nan)
            ceil = float(np.sqrt(ra * rb)) if np.isfinite(ra) and np.isfinite(rb) else np.nan
            rows.append({"arm_a": a, "arm_b": b, "cosine": cos, "ceiling": ceil,
                         "cosine_corrected": cos / ceil if ceil and np.isfinite(ceil) else np.nan})
    return pd.DataFrame(rows)


def weighting_decomposition(embedded_all: pd.DataFrame, arm_a: str, arm_b: str, *,
                            seed: int = BOOT_SEED) -> pd.DataFrame:
    """Is the cross-method divergence about the LOSS or about the DATA?

    The as-trained cosine between PTO's and GRPO's update directions confounds two differences:
    they see different candidate **pools** (one branches a greedy trunk, the other samples a group
    off a fixed prompt) and they apply different weighting **rules**. Every group logs all of its
    candidates' scores, so the two can be separated with no new data — hold the groups fixed and
    swap only the rule (:func:`reweight`):

    ==========================================  ==========================================
    ``as trained``                              rule *and* data differ — the headline cosine
    ``same data, other rule`` (one row per arm) the RULE's effect, on that arm's own pool
    ``same rule, data differs`` (dpo and grpo)  the DATA's effect, rule held constant
    ==========================================  ==========================================

    If the same-rule rows stay as low as ``as trained`` while the same-data rows are high, the two
    methods diverge because of the candidates they generate, not because of how they weight them —
    and "PTO vs GRPO" is then a statement about exploration, not about DPO vs group-relative PPO.
    The reverse pattern says the loss family is doing the work.

    Takes the embedded frame with **every** candidate kept
    (``load_weighted_candidates(..., drop_zero_weight=False)`` → :func:`embed_candidates`), since a
    counterfactual rule needs the candidates the native rule discarded. Each row carries the
    attenuation ceiling for its own pair of weightings, because these directions are not equally
    well estimated (a DPO-style ±1 direction uses 2 candidates per group, a GRPO-style one uses 8).

    ⚠ The ``read`` column says which cosine to quote, and it is not cosmetic. The attenuation
    correction assumes the two estimates' errors are **independent** — true across arms, false for
    the same-data rows, where both directions come from the very same groups and share their noise.
    There the correction over-corrects (it can exceed 1.0, which is the tell), so read the RAW
    cosine on those rows and the corrected one on the cross-arm rows.
    """
    if embedded_all.empty:
        return pd.DataFrame()
    variants = {"native": embedded_all, "dpo": reweight(embedded_all, "dpo"),
                "grpo": reweight(embedded_all, "grpo")}
    dirs, rel = {}, {}
    for name, frame in variants.items():
        f = frame[frame["weight"] != 0]
        if f.empty:
            continue
        d = direction_by_arm(f)
        q = pooled_direction_quality(f, d, seed=seed).set_index("arm")["split_half_cos"]
        for arm, vec in d.items():
            dirs[(name, arm)] = vec
            rel[(name, arm)] = _spearman_brown(q.get(arm, np.nan))
    plan = [
        ("as trained (rule AND data differ)", ("native", arm_a), ("native", arm_b)),
        (f"same data ({arm_a}), rule swapped", ("native", arm_a), ("grpo", arm_a)),
        (f"same data ({arm_b}), rule swapped", ("native", arm_b), ("dpo", arm_b)),
        ("same rule (group-relative), data differs", ("grpo", arm_a), ("native", arm_b)),
        ("same rule (best-vs-worst), data differs", ("native", arm_a), ("dpo", arm_b)),
    ]
    rows = []
    for label, ka, kb in plan:
        if ka not in dirs or kb not in dirs:
            continue
        cos = float(dirs[ka] @ dirs[kb])
        ra, rb = rel.get(ka, np.nan), rel.get(kb, np.nan)
        ceil = float(np.sqrt(ra * rb)) if np.isfinite(ra) and np.isfinite(rb) else np.nan
        same_arm = ka[1] == kb[1]
        rows.append({"comparison": label, "a": f"{ka[1]}({ka[0]})", "b": f"{kb[1]}({kb[0]})",
                     "cosine": cos, "ceiling": ceil,
                     "cosine_corrected": cos / ceil if ceil and np.isfinite(ceil) else np.nan,
                     "read": "cosine (same groups — see note)" if same_arm else "cosine_corrected"})
    return pd.DataFrame(rows)


def rule_reconstruction_check(cands_all: pd.DataFrame, arm_label: str) -> dict:
    """Does the score-only ``dpo`` rule pick the same candidates PTO's trainer recorded?

    Reports the honest version of that question. Exact row agreement understates it: ~36% of PTO
    groups have **tied** maxima, and a tie-break is arbitrary by definition. So this returns both
    ``picks_a_maximum`` (does the rule select a top-scoring candidate at all — the property that
    actually has to hold) and ``tie_rate`` (how often the choice among equals was free).
    """
    d = cands_all[(cands_all["arm"] == arm_label) & cands_all["role"].isin(["chosen", "rejected"])]
    if d.empty:
        return {}
    g = cands_all[cands_all["arm"] == arm_label].groupby(_GROUP_KEYS)["score"]
    hi, lo = g.transform("max"), g.transform("min")
    ch, rj = d[d.role == "chosen"], d[d.role == "rejected"]
    n_at_max = cands_all[cands_all["arm"] == arm_label].assign(
        _m=lambda x: x["score"] == hi).groupby(_GROUP_KEYS)["_m"].transform("sum")
    return {"arm": arm_label,
            "chosen_picks_a_maximum": float((ch["score"] == hi.loc[ch.index]).mean()),
            "rejected_picks_a_minimum": float((rj["score"] == lo.loc[rj.index]).mean()),
            "tie_rate_at_max": float((n_at_max.loc[ch.index] > 1).mean()),
            "n_groups": int(len(ch))}


def direction_agreement_with_pairs(embedded: pd.DataFrame, arm_label: str,
                                   pairs_directions: dict) -> pd.DataFrame:
    """Sanity gate: the candidate-derived PTO direction vs the one built from ``pairs.csv``.

    Two independent files describe the same DPO update — ``generations.jsonl`` (roles per
    candidate) and ``pairs.csv`` (the emitted pairs). Cosine ≈ 1 per iteration says the new
    method-agnostic path reproduces the original PTO probe, which is what licenses reading the
    GRPO side the same way. Anything well below 1 means the two logs disagree about what was
    trained on — a data-integrity finding, not a plotting detail.
    """
    dirs = direction_by_iter(embedded[embedded["arm"] == arm_label])
    own = dirs.get(arm_label, {})
    return pd.DataFrame([{"train_iter": i, "cosine": float(own[i] @ pairs_directions[i])}
                         for i in sorted(set(own) & set(pairs_directions))])


# ── The lexical half: exact, needs no embeddings, covers every group ──────────────
_LEX_FEATURES = {"len_chars": "w_len", "n_questions": "w_question",
                 "affirm_marker": "w_affirm", "overpraise_marker": "w_overpraise"}


def weighted_lexical_contrast(cands: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, iter): the feature difference the update pushes for, in chosen−rejected units.

    ``Σ_g w · f`` averaged over groups. Because weights are normalized to ``Σ|w| = 2``, a value of
    ``+40`` on ``w_len`` reads as "the update pushes toward completions ~40 characters longer",
    identically for DPO's pair and for GRPO's advantage-weighted group.

    Exact and cheap — it uses every group (no embedding, no sampling), so it is the right series to
    correlate against outcomes; the embedding direction is the semantic complement.

    Each feature also gets a ``<name>_se`` — the standard error over groups. These per-pair pushes
    are small numbers (one PTO pair moves the affirmation-marker rate by ~0.05), so the SE is what
    separates "the update leans this way" from "this iteration's groups happened to differ".
    """
    if cands.empty:
        return pd.DataFrame()
    out = []
    for (arm, it), g in cands.groupby(["arm", "train_iter"]):
        contrib = {f"_{name}": g["weight"] * g[src].astype(float)
                   for src, name in _LEX_FEATURES.items()}
        per_group = g.assign(**contrib).groupby(_GROUP_KEYS)[list(contrib)].sum()
        row = {"arm": arm, "method": g["method"].iloc[0], "K": g["K"].iloc[0],
               "train_iter": int(it), "n_groups": len(per_group), "n_candidates": len(g)}
        for name in _LEX_FEATURES.values():
            row[name] = float(per_group[f"_{name}"].mean())
            row[f"{name}_se"] = float(per_group[f"_{name}"].sem())
        out.append(row)
    return pd.DataFrame(out).sort_values(["arm", "train_iter"]).reset_index(drop=True)


def pool_mean_by_iter(cands_all: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, iter): the mean feature over ALL candidates — what the policy *generates*.

    The other half of the drift question. :func:`weighted_lexical_contrast` measures what the
    update **selects for** within a group; this measures what the policy **produces** in the first
    place. They answer different questions and can move independently:

    - pool rising, selection ≈ 0 → the policy drifted on its own and the reward is merely
      following it (the update is not pulling);
    - pool flat, selection > 0 → the reward is pulling against the policy's own tendency;
    - both rising → the pull and the drift compound, which is the reward-hacking story in full.

    Needs the UNFILTERED frame — ``load_weighted_candidates(..., drop_zero_weight=False)`` — or the
    "pool" is just the two extremes DPO kept, which is exactly the thing being controlled for.
    Columns mirror the contrast table with a ``pool_`` prefix (``pool_len``, ``pool_question``, …).

    **Iteration indexing.** ``train_iter`` *n* samples its candidates from π\\ :sub:`n-1` (the
    iter-start policy), so the pool at *n* describes the same policy the eval set calls
    ``model_iter_{n-1}`` — off by one from the selection contrast on the same row, which describes
    the update *applied* at *n*. That is intentional and is exactly the pairing the question needs:
    what the policy already produced, next to what the update did about it.
    """
    if cands_all.empty:
        return pd.DataFrame()
    rows = []
    for (arm, it), g in cands_all.groupby(["arm", "train_iter"]):
        row = {"arm": arm, "method": g["method"].iloc[0], "K": g["K"].iloc[0],
               "train_iter": int(it), "n_candidates": len(g)}
        for src, name in _LEX_FEATURES.items():
            v = g[src].astype(float)
            row[name.replace("w_", "pool_")] = float(v.mean())
            row[name.replace("w_", "pool_") + "_se"] = float(v.sem())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "train_iter"]).reset_index(drop=True)


def pair_yield_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iter): how many groups actually produced a gradient, out of how many were built.

    The asymmetry this exposes is structural, not cosmetic. GRPO trains on **every** prompt it
    builds; PTO emits a pair only where the best and worst branch differ by more than ``τ``, so its
    usable signal can dry up as the policy's branches converge — and a shrinking pair count is a
    candidate explanation for a flattening learning curve that no outcome figure can see.

    Columns: ``groups_built`` (every branch point / prompt group logged for that iteration),
    ``groups_trained`` (those with both an up- and a down-weighted side), ``yield_rate``, and
    ``mean_margin`` / ``median_margin`` — the best−worst score gap, i.e. how decisive the surviving
    groups were.
    """
    gens = load_generations(arms)
    if gens.empty:
        return pd.DataFrame()
    d = gens[(gens["phase"] != "eval") & gens["score"].notna()].copy()
    d["epoch"] = d["epoch"].fillna(-1.0)
    trained = load_weighted_candidates(arms, drop_zero_weight=True)
    trained_keys = set(map(tuple, trained[_GROUP_KEYS].to_numpy())) if not trained.empty else set()
    rows = []
    for (arm, it), g in d.groupby(["arm", "train_iter"]):
        keys = g.groupby(_GROUP_KEYS)
        built = keys.ngroups
        margins = keys["score"].max() - keys["score"].min()
        n_trained = sum(1 for k in keys.groups if tuple(k) in trained_keys)
        rows.append({"arm": arm, "method": g["method"].iloc[0], "train_iter": int(it),
                     "groups_built": built, "groups_trained": n_trained,
                     "yield_rate": n_trained / built if built else np.nan,
                     "mean_margin": float(margins.mean()),
                     "median_margin": float(margins.median())})
    return pd.DataFrame(rows).sort_values(["arm", "train_iter"]).reset_index(drop=True)


def pref_examples(cands: pd.DataFrame, *, arm: str, iters: Optional[Sequence[int]] = None,
                  k: int = 3, max_chars: int = 320) -> pd.DataFrame:
    """The most decisive up- vs down-weighted completions, as text — early vs late.

    Every other artifact here is an aggregate; a reader (and a thesis chapter) also wants to see
    what the drift actually looks like in words. Takes the ``k`` groups per iteration with the
    largest score gap — **deliberately not a random sample**: these are the groups carrying most of
    the update, and they are the ones worth reading. Quote them as illustration, never as evidence
    of a rate.
    """
    if cands.empty:
        return pd.DataFrame()
    d = cands[(cands["arm"] == arm) & (cands["weight"] != 0)]
    if iters is not None:
        d = d[d["train_iter"].isin(list(iters))]
    if d.empty:
        return pd.DataFrame()
    rows = []
    for it, g in d.groupby("train_iter"):
        gg = g.assign(_gid=g.groupby(_GROUP_KEYS).ngroup())
        gaps = gg.groupby("_gid")["score"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
        for gid in gaps.head(k).index:
            grp = gg[gg["_gid"] == gid]
            hi, lo = grp.loc[grp["weight"].idxmax()], grp.loc[grp["weight"].idxmin()]
            clip = lambda t: (str(t)[:max_chars] + "…") if len(str(t)) > max_chars else str(t)
            rows.append({"arm": arm, "train_iter": int(it),
                         "score_gap": round(float(hi["score"] - lo["score"]), 3),
                         "up_weighted": clip(hi["completion"]).replace("\n", " "),
                         "down_weighted": clip(lo["completion"]).replace("\n", " ")})
    return pd.DataFrame(rows)


_LEX_TITLES = {"w_len": "completion length (chars)", "w_question": "question marks",
               "w_affirm": "affirmation marker", "w_overpraise": "over-praise marker"}


def plot_lexical_push(lex: pd.DataFrame, *, features: Optional[List[str]] = None,
                      palette: Optional[dict] = None, ncols: int = 4):
    """What each iteration's update pushes toward, per arm — the cross-method figure.

    One panel per lexical feature; y is the per-group weighted contrast (± 1 SE over groups), so
    **0 = the update is indifferent** to that feature and positive = it pushes toward more of it.
    Both methods appear on the same axes because the weights were normalized to a shared scale
    (see :func:`load_weighted_candidates`).
    """
    import matplotlib.pyplot as plt
    from .plotting_style import grid
    if lex.empty:
        return None
    features = [f for f in (features or list(_LEX_FEATURES.values())) if f in lex.columns]
    fig, axes = grid(len(features), ncols=min(ncols, len(features)), panel=(4.2, 3.0))
    for ax, f in zip(axes, features):
        for arm, g in lex.sort_values("train_iter").groupby("arm"):
            se = g[f"{f}_se"] if f"{f}_se" in g else None
            ax.errorbar(g["train_iter"], g[f], yerr=se, marker="o", ms=4, capsize=2, lw=1.4,
                        label=arm, color=(palette or {}).get(arm))
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.set_title(_LEX_TITLES.get(f, f), fontsize=9)
        ax.set_xlabel("training iteration"); ax.set_ylabel(f"{f}  (per group, ±1 SE)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("What the update pushes toward, per iteration — both methods on one scale",
                 y=1.04, fontweight="bold")
    fig.text(0.5, -0.02, "Weighted contrast Σ w·feature per group (DPO's ±1 pair scale; GRPO's "
                         "standardized advantages rescaled to match). 0 = indifferent.",
             ha="center", va="top", fontsize=7.5, style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def plot_selection_vs_generation(pool: pd.DataFrame, lex: pd.DataFrame, *,
                                 features: Optional[List[str]] = None,
                                 palette: Optional[dict] = None):
    """Two rows: what the policy GENERATES (pool mean) over what the update SELECTS for.

    Reading the columns top-to-bottom answers "is the reward pulling the policy toward this, or has
    the policy already gone there on its own?" — the question that separates a reward that *causes*
    the drift from one that merely *ratifies* it.
    """
    import matplotlib.pyplot as plt
    if pool.empty or lex.empty:
        return None
    features = [f for f in (features or list(_LEX_FEATURES.values())) if f in lex.columns]
    n = len(features)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.2), squeeze=False)
    for j, f in enumerate(features):
        pf = f.replace("w_", "pool_")
        for arm, g in pool.sort_values("train_iter").groupby("arm"):
            axes[0][j].errorbar(g["train_iter"], g[pf], yerr=g.get(f"{pf}_se"), marker="o", ms=4,
                                capsize=2, lw=1.4, label=arm, color=(palette or {}).get(arm))
        for arm, g in lex.sort_values("train_iter").groupby("arm"):
            axes[1][j].errorbar(g["train_iter"], g[f], yerr=g.get(f"{f}_se"), marker="o", ms=4,
                                capsize=2, lw=1.4, label=arm, color=(palette or {}).get(arm))
        axes[1][j].axhline(0, color="grey", lw=0.7, ls="--")
        axes[0][j].set_title(_LEX_TITLES.get(f, f), fontsize=9)
        axes[0][j].set_ylabel("GENERATED\n(mean over all candidates)" if j == 0 else "")
        axes[1][j].set_ylabel("SELECTED FOR\n(weighted contrast)" if j == 0 else "")
        axes[1][j].set_xlabel("training iteration")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("Generation vs selection — does the update pull the drift, or follow it?",
                 y=1.02, fontweight="bold")
    fig.text(0.5, -0.01, "Top: what the policy produces (unweighted mean over every candidate). "
                         "Bottom: what the update pushes toward within a group (0 = indifferent).",
             ha="center", va="top", fontsize=7.5, style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def plot_pair_yield(yield_df: pd.DataFrame, *, palette: Optional[dict] = None):
    """How much of each iteration's built signal survived to train — and how decisive it was."""
    import matplotlib.pyplot as plt
    if yield_df.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    panels = [("groups_trained", "groups that trained"),
              ("yield_rate", "yield = trained / built"),
              ("mean_margin", "best−worst score gap")]
    for ax, (col, title) in zip(axes, panels):
        for arm, g in yield_df.sort_values("train_iter").groupby("arm"):
            ax.plot(g["train_iter"], g[col], marker="o", ms=4, lw=1.5, label=arm,
                    color=(palette or {}).get(arm))
        ax.set_title(title, fontsize=9); ax.set_xlabel("training iteration"); ax.set_ylabel(col)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("How much usable training signal each iteration actually produced",
                 y=1.05, fontweight="bold")
    fig.text(0.5, -0.02, "GRPO trains on every prompt it builds; PTO emits a pair only where the "
                         "best and worst branch differ by more than tau, so its yield can fall as "
                         "branches converge.", ha="center", va="top", fontsize=7.5, style="italic",
             color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def preference_features_by_iter(cands: pd.DataFrame, *, directions: Optional[dict] = None,
                                quality: Optional[pd.DataFrame] = None,
                                categories: Optional[dict] = None) -> pd.DataFrame:
    """One row per (arm, train_iter): everything about that iteration's update, ready to correlate.

    Merges the exact lexical contrasts (:func:`weighted_lexical_contrast`) with, when
    ``directions`` is given, each MI concept's projection onto that iteration's direction
    (``cat_<Concept>``) and ``dir_cos_prev`` (cosine with the previous iteration's direction — is
    the target holding still or re-aiming?), plus ``wins_correct``/``split_half_cos`` from
    ``quality``. This is the left-hand side of :func:`link_to_outcomes`.
    """
    feats = weighted_lexical_contrast(cands)
    if feats.empty:
        return feats
    if directions:
        cat_rows = []
        for arm, dirs in directions.items():
            cat = category_projection(dirs, categories)
            if cat.empty:
                continue
            wide = cat.pivot_table(index="train_iter", columns="category", values="score")
            wide.columns = [f"cat_{c}" for c in wide.columns]
            wide = wide.reset_index().assign(arm=arm)
            its = sorted(dirs)
            wide["dir_cos_prev"] = wide["train_iter"].map(
                {i: (float(dirs[i] @ dirs[p]) if (p := i - 1) in dirs else np.nan) for i in its})
            cat_rows.append(wide)
        if cat_rows:
            feats = feats.merge(pd.concat(cat_rows, ignore_index=True),
                                on=["arm", "train_iter"], how="left")
    if quality is not None and not quality.empty:
        feats = feats.merge(quality[["arm", "train_iter", "wins_correct", "split_half_cos"]],
                            on=["arm", "train_iter"], how="left")
    return feats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   TRAINING SIGNAL -> EVAL MOVE — does what the update prefers predict the gain? ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

FEATURE_COLS = ["w_len", "w_question", "w_affirm", "w_overpraise",
                "cat_Affirmation", "cat_Reflection", "cat_OpenQuestion", "cat_ChangeTalk",
                "cat_SustainTalk", "cat_TherapistActions", "dir_cos_prev", "wins_correct"]


def link_to_outcomes(features: pd.DataFrame, scores_long: pd.DataFrame, *,
                     metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Join each iteration's preference features to the eval delta that iteration's update caused.

    The iteration bookkeeping is the whole point and is easy to get wrong: the update performed in
    **train_iter n** produced adapter ``iteration_n``, whose conversations are ``model_iter_n``.
    So its effect is ``eval(model_iter_n) − eval(model_iter_{n-1})``, persona-paired over the 96
    shared personas via :func:`~eda_analysis.stats.compare_two_models` (NOT a difference of
    iteration means — the personas are reshuffled every iteration).

    Long format: one row per (arm, train_iter, metric) carrying every feature column plus
    ``delta_mean`` / ``dz`` / ``p`` for that step. Iterations whose two model states are not both
    scored are dropped.

    ⚠ Correlational and small-*n* (≤10 steps per arm). It cannot separate "the update pushed
    affirmation, which raised the score" from "iterations where the policy was improving anyway
    also had affirmation-heavy branches" — read it as a mechanism *consistent with* the outcome
    curves, never as the cause of them.
    """
    from .stats import compare_two_models
    if features.empty or scores_long.empty:
        return pd.DataFrame()
    rows = []
    for arm, f in features.groupby("arm"):
        sub = scores_long[scores_long["arm"] == arm]
        if sub.empty:
            continue
        model_at = {int(i): g["model"].iloc[0] for i, g in sub.groupby("iteration")}
        for r in f.itertuples(index=False):
            it = int(r.train_iter)
            if it not in model_at or (it - 1) not in model_at:
                continue
            cmp = compare_two_models(sub, model_at[it], model_at[it - 1], metrics)
            if cmp.empty:
                continue
            base = {k: v for k, v in r._asdict().items()}
            for c in cmp.itertuples(index=False):
                rows.append({**base, "metric": c.metric, "delta_mean": c.mean_delta,
                             "dz": c.dz, "p": c.p, "n_paired": c.n})
    return pd.DataFrame(rows)


def _partial_spearman(x, y, z):
    """Spearman correlation of *x* and *y* with *z* partialled out (rank residuals)."""
    from scipy import stats as _st
    rx, ry, rz = (_st.rankdata(v) for v in (x, y, z))
    zc = rz - rz.mean()
    denom = float((zc ** 2).sum())
    if denom <= 0:
        return np.nan, np.nan
    res = []
    for r in (rx, ry):
        rc = r - r.mean()
        res.append(rc - (float((rc * zc).sum()) / denom) * zc)
    if np.allclose(res[0], 0) or np.allclose(res[1], 0):
        return np.nan, np.nan
    stat = _st.pearsonr(res[0], res[1])
    return float(stat[0]), float(stat[1])


def outcome_correlations(link: pd.DataFrame, *, features: Optional[List[str]] = None,
                         min_n: int = 4) -> pd.DataFrame:
    """Does a feature of the update predict the eval move — beyond both just drifting with time?

    Three numbers per (scope, metric, feature):

    - ``spearman_rho`` — the raw association across iterations.
    - ``rho_feature_vs_iter`` — how strongly the feature itself trends with iteration.
    - ``rho_partial_iter`` — **the one to read**: Spearman with ``train_iter`` partialled out of
      both sides.

    The partial is not optional decoration here. Most of these features rise monotonically over
    training and most eval deltas shrink (gains taper) or grow (MICI), so *any* monotone feature
    correlates with *any* monotone delta — the raw ρ is confounded with iteration index almost by
    construction. A raw ρ that collapses once ``train_iter`` is removed is a shared time trend, not
    a mechanism; one that survives means the iterations that pushed harder moved further **relative
    to where they sat in training**.

    Reported per arm and, because a single arm gives at most 10 points, pooled per method
    (``scope`` = arm label or ``"<METHOD> (pooled)"``). Rows with fewer than ``min_n`` iterations
    are dropped rather than shown with an uninterpretable ρ.

    **Descriptive only** — n ≤ 10 per arm, no multiplicity correction across the feature × metric
    grid, and the caveat in :func:`link_to_outcomes` applies to every row.
    """
    from scipy import stats as _st
    if link.empty:
        return pd.DataFrame()
    feats = [c for c in (features or FEATURE_COLS) if c in link.columns]
    scopes = [(a, d) for a, d in link.groupby("arm")]
    scopes += [(f"{m} (pooled)", d) for m, d in link.groupby("method") if d["arm"].nunique() > 1]
    rows = []
    for scope, d in scopes:
        for metric, dm in d.groupby("metric"):
            for f in feats:
                x = dm[[f, "delta_mean", "train_iter"]].dropna()
                if len(x) < min_n or x[f].nunique() < 3:
                    continue
                rho, p = _st.spearmanr(x[f], x["delta_mean"])
                rho_p, p_p = _partial_spearman(x[f], x["delta_mean"], x["train_iter"])
                trend, _ = _st.spearmanr(x[f], x["train_iter"])
                rows.append({"scope": scope, "metric": metric, "feature": f, "n_iters": len(x),
                             "spearman_rho": float(rho), "p": float(p),
                             "rho_partial_iter": rho_p, "p_partial": p_p,
                             "rho_feature_vs_iter": float(trend)})
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "feature", "scope"]).reset_index(drop=True) if not out.empty else out


def plot_pref_outcome(link: pd.DataFrame, *, feature: str, metrics: Optional[List[str]] = None,
                      palette: Optional[dict] = None, ncols: int = 3):
    """Scatter of one preference feature against the eval delta it preceded, one panel per metric.

    Points are iterations, coloured by arm, annotated with the point's iteration number so an
    outlier can be traced back to a specific update; the dashed line is the per-arm least-squares
    fit and the corner text is Spearman ρ per arm.
    """
    import matplotlib.pyplot as plt
    from .constants import display_label
    from .plotting_style import grid
    if link.empty or feature not in link.columns:
        return None
    metrics = metrics or [m for m in ("Q1Q2", "MICI", "MITI") if m in set(link["metric"])]
    metrics = [m for m in metrics if m in set(link["metric"])]
    if not metrics:
        return None
    fig, axes = grid(len(metrics), ncols=min(ncols, len(metrics)), panel=(4.6, 3.4))
    for ax, m in zip(axes, metrics):
        d = link[link["metric"] == m].dropna(subset=[feature, "delta_mean"])
        notes = []
        for arm, g in d.groupby("arm"):
            col = (palette or {}).get(arm, None)
            ax.scatter(g[feature], g["delta_mean"], s=34, label=arm, color=col, zorder=3)
            for r in g.itertuples(index=False):
                ax.annotate(str(int(r.train_iter)), (getattr(r, feature), r.delta_mean),
                            textcoords="offset points", xytext=(4, 3), fontsize=6, color="#555555")
            if len(g) >= 3 and g[feature].nunique() >= 2:
                b, a = np.polyfit(g[feature], g["delta_mean"], 1)
                xs = np.linspace(g[feature].min(), g[feature].max(), 20)
                ax.plot(xs, a + b * xs, ls="--", lw=1.0, color=col, alpha=0.8, zorder=2)
                from scipy import stats as _st
                notes.append(f"{arm}: ρ={_st.spearmanr(g[feature], g['delta_mean'])[0]:+.2f}")
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(display_label(m), fontsize=9)
        ax.set_xlabel(feature); ax.set_ylabel(f"Δ {display_label(m)} vs prev iter")
        if notes:
            ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, fontsize=6.5,
                    va="bottom", color="#333333")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Training-signal feature '{feature}' vs the eval move it preceded",
                 y=1.03, fontweight="bold")
    fig.text(0.5, -0.02, "Each point is one training iteration (label = iteration). Correlational, "
                         "n <= 10 per arm — a mechanism consistent with the curves, not a cause.",
             ha="center", va="top", fontsize=7.5, style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def plot_category_compare(cat_by_arm: dict, *, palette: Optional[dict] = None,
                          title: str = "MI-concept preference by arm"):
    """Overlay the MI-concept drift of several arms — one panel per concept, one line per arm.

    Takes ``{arm_label: category_projection frame}``. Used for both cross-arm reads the notebook
    makes (PTO vs GRPO, K=0 vs K=5) so neither is a hand-rolled inline plot.
    """
    import matplotlib.pyplot as plt
    frames = {a: c for a, c in cat_by_arm.items() if c is not None and not c.empty}
    if len(frames) < 2:
        return None
    cats = sorted(set.intersection(*(set(c["category"]) for c in frames.values())))
    if not cats:
        return None
    fig, axes = plt.subplots(1, len(cats), figsize=(2.8 * len(cats), 3.3), squeeze=False,
                             sharey=True)
    for ax, c in zip(axes.flat, cats):
        for arm, cat in frames.items():
            d = cat[cat["category"] == c].sort_values("train_iter")
            ax.plot(d["train_iter"], d["score"], marker="o", label=arm,
                    color=(palette or {}).get(arm))
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_title(c, fontsize=8); ax.set_xlabel("train iter")
    axes.flat[0].legend(fontsize=7, frameon=False); axes.flat[0].set_ylabel("projection")
    fig.suptitle(title, y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig
