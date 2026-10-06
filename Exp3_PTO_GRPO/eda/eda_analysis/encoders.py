"""
encoders.py — the GRPO update direction in five sentence-embedding spaces, read through sentences
and words (family ``lookahead/mechanism``, section 1d).

Ported from the exploratory ``meetings/build/embedding_directions_2026-10-06.py`` (its write-up:
``meetings/2026-10-06_embedding_directions/RESULTS.md``) so that the paper can cite EDA-owned tables.
Of that script's variants only ONE is ported, the one the paper's update-direction proxy uses:

* **Estimator** — :mod:`pref`'s, unchanged: each candidate weighted by its standardised
  group-relative advantage, rescaled to ``Σ|w| = 2`` per gradient group
  (:func:`pref.load_weighted_candidates`), direction = ``normalize(Σ w · emb)`` per (run,
  train_iter); noise-corrected cosines from 50 conversation split-halves
  (:func:`pref.k_direction_cosines_by_iter`, :func:`pref.direction_stability_by_iter`). The meetings
  script calls this ``advw``; its ``winlose`` (Doron's mean of best − worst) is not ported.
* **Candidates** — ALL of them (the meetings ``all`` set): no leaked-marker or degenerate-text
  filter, here or in the category and log-odds readouts.

Five encoders (:data:`ENCODERS`): all-MiniLM-L6-v2 (the paper's), gte-base-en-v1.5,
mxbai-embed-large-v1, Qwen3-Embedding-0.6B, and Llama-3.2-1B — the therapist's own base — read at
the output of decoder layer 8 of 16 (before the final norm), mean-pooled over the text's tokens with
BOS excluded. This module is CPU-only: it READS the per-encoder caches
``eda/.emb_cache/encoders/<key>/{texts.json.gz, emb.npy}`` (texts keyed by the exact string, rows
unit-normalised) that the GPU step ``tools/embed_encoders.py`` writes once. ``minilm``'s cache is
seeded from the EDA's own MiniLM vectors (``eda/.emb_cache/all-MiniLM-L6-v2.pkl``, float32), so its
rows reproduce ``direction_k_by_iter_grpo`` / ``direction_stability_grpo`` exactly; the other four
are float16, renormalised on load.

Three readouts of the same per-(run, train_iter) direction:

1. **Cosines** (:func:`encoder_cosines`) — K=0 vs K=5 at each iteration and each run's iteration n
   vs n−1, raw and noise-corrected, per encoder. Cosines are NOT comparable ACROSS encoders (each
   space has its own geometry); agreement between encoders is about the pattern along the run.
2. **Sentence categories** (:func:`category_scores`) — Doron's sentence pool
   (:func:`sentence_pool`) projected on each direction; per category, ``z`` = (mean projection of
   the category's pool sentences − mean of the other pool sentences) / SD of the pool's projections,
   i.e. how far toward the direction the category sits, in pool SDs. 95% interval from conversation
   bootstraps; the same draws for both runs, so the K0 − K5 difference is paired.
3. **Words, embedding-free** (:func:`lexical_logodds`) — Monroe, Colaresi & Quinn (2008) log-odds
   with an informative Dirichlet prior of word counts in each group's best vs worst candidate. The
   one to QUOTE is :func:`lexical_logodds_clean` — the exception to "all candidates": it drops
   candidates with a leaked chat marker or degenerate text (the meetings definitions, copied
   verbatim) and re-chooses best and worst among the rest, because on all candidates the losing
   side is single-letter fragments of degenerate text and marker pieces (``im``, ``end``) rank
   among K=0's late winners.

Caveats (keep them beside any number quoted from these tables):

* **Per-group directions are weak.** A cell's direction orders a single held-out group's best
  above its worst only modestly more often than chance (exploratory, NOT an EDA table:
  ``meetings/2026-09-29_doron_grpo_groups/directions/RESULTS.md`` — mean 0.693 for K=0, falling to
  0.516 for K=5 at iteration 10, gte embeddings); everything here is a CELL average over hundreds
  of groups. Read ``corrected`` only where ``rel_a`` and ``rel_b`` are well above 0 — late K=5 cells
  fall to split-half agreement near 0.15 in MiniLM, where the correction divides by noise.
* **From train_iter 2 on the two runs sample from different policies,** so a cell's direction
  mixes what each reward prefers with what each policy proposes; only train_iter 1 (both branch
  from the Base) is a same-policy contrast.
* **The categories are regexes** written before the run (see :data:`CATEGORIES`), matched on the
  pool sentence; a sentence can fall in several, and ``advice`` / ``reflection opener`` are small.
* **Isolated-word embeddings are NOT ported.** The meetings script also embedded single words
  (wordfreq's top 20k and the corpus vocabulary) and projected them on the directions; that gave
  topic words and noise in every encoder (mostly noise for Llama), so the word readout here is the
  embedding-free log-odds only.

Conventions: functions take frames and return tidy frames / figures, nothing writes to disk (the
notebook exports), bootstraps seed with :data:`constants.BOOT_SEED`.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import BOOT_SEED, k_of
from . import pref as _pref

__all__ = [
    "ENCODERS", "ENCODER_CACHE_DIR", "TOOL_CMD", "CATEGORIES", "EncoderCacheMissing",
    "cache_status", "required_texts", "require_caches", "load_encoder_embeddings",
    "embedded_for", "conversation_sums", "encoder_cosines", "sentence_pool",
    "category_scores", "category_summary", "top_sentences", "lexical_logodds", "LEAK", "POOL_MARKER",
    "degenerate_text", "clean_mask", "clean_counts", "lexical_logodds_clean", "plot_encoder_categories",
]

#: Encoder key -> label, in table / legend order. The GPU-side spec (model id, pooling, dtype,
#: batch sizes) lives in ``tools/embed_encoders.py``, the only place a model is ever loaded.
ENCODERS: Dict[str, str] = {
    "minilm": "all-MiniLM-L6-v2",
    "gte": "gte-base-en-v1.5",
    "mxbai": "mxbai-embed-large-v1",
    "qwen3": "Qwen3-Embedding-0.6B",
    "llama8": "Llama-3.2-1B layer 8 (mean-pooled, BOS excluded)",
}
ENCODER_CACHE_DIR = os.path.join(_pref._CACHE_DIR, "encoders")
TOOL_CMD = r".venv\Scripts\python.exe tools/embed_encoders.py   (from Exp3_PTO_GRPO/eda; GPU, run once)"

# Sentence categories, fixed on 2026-10-06 BEFORE this run (informed by the 10-05 reads and by the
# utterance coder's code families: praise / thanks / agreement ~ PRA, question ~ OQ+CQ, reflection
# opener ~ SR+CR, advice ~ PERS+GI, list preamble = the templated-list artefact). Matched on the
# pool sentence, case-insensitive; a sentence may fall in several.
# (Copied verbatim from meetings/build/embedding_directions_2026-10-06.py.)
CATEGORIES = {
    "praise": re.compile(
        r"\bproud of you|\bthrilled\b|\bso (glad|happy|excited)\b|\bamazing\b|\bincredible\b|\bwonderful\b|"
        r"\bfantastic\b|\bcongratulat\w*|\bcourage(ous)?\b|\bbrave\b|\binspir\w*|\bwell done\b|"
        r"\bgreat (job|work|progress|to hear)\b|\byou('re| are) (doing )?(great|so strong|strong|awesome)\b", re.I),
    "thanks": re.compile(r"\bthank(s| you)\b|\bappreciate\b|\bgrateful\b", re.I),
    "agreement": re.compile(r"^\W*(absolutely|exactly|definitely|of course|you'?re (absolutely )?right|i agree|"
                            r"that'?s (exactly |absolutely )?(right|true))\b", re.I),
    "question": re.compile(r"\?\W*$"),
    "reflection opener": re.compile(r"\b(it sounds like|sounds like you|it seems (like|that)|you('re| are) feeling|"
                                    r"you feel|what i('m| am) hearing|you mentioned|you said|so you('re| are))\b", re.I),
    "advice": re.compile(r"\b(you (should|could|might|can) (try|consider|start)|try to|one tip|tips?|i (suggest|recommend)|"
                         r"start (with|by)|set (small|realistic|achievable|specific)|remember,|make sure)\b", re.I),
    "list preamble": re.compile(r"\b(here are|here they are|as follows|the (\w+ )?(steps|questions|strategies|tips|goals|"
                                r"actions|examples|areas|features|considerations) are)\b", re.I),
}

_ARM_A, _ARM_B = "GRPO_LA0", "GRPO_LA5"
_CONV = ["arm", "train_iter", "conversation_id"]
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")          # Doron's sentence splitter
_WORDTOK = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ── the exploratory "clean" candidate filter, copied VERBATIM (used only by lexical_logodds_clean) ──
# LEAK is check_doron_directions_2026-10-05.py's ``LEAK = rd.FEATURES["leaked chat marker"]``, i.e. the regex of
# analyze_grpo_round_directions_2026-09-29.py's FEATURES (the same pattern as pref._RE_MALFORMED). _NON_ASCII, _WORD
# and degenerate_text() are copied from meetings/build/analyze_grpo_round_directions_2026-09-29.py unchanged. The
# meetings run applied degenerate_text to every candidate of both GRPO runs at once — its vocabulary (words used in
# >= 200 candidates) depends on that set, so pass the same frame.
LEAK = re.compile(r"<\|?im_")
_NON_ASCII = re.compile("[^\x00-\x7f‘’“”–—…•]")   # curly quotes etc. are fine
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")

#: The sentence pool's chat-marker filter: LEAK widened by any ``|im`` (``</|im_end|>`` and a bare
#: trailing ``<|im`` slip past LEAK's ``<|?im_``). Used only by :func:`sentence_pool`.
POOL_MARKER = re.compile(r"<\|?im_|\|im")


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


class EncoderCacheMissing(LookupError):
    """An encoder cache is absent, or lacks some of the texts asked for. The message names the tool."""


# ── cache access ──────────────────────────────────────────────────────────────
def _cache_paths(key: str):
    d = os.path.join(ENCODER_CACHE_DIR, key)
    return d, os.path.join(d, "texts.json.gz"), os.path.join(d, "emb.npy")


def _open(key: str):
    """``(pd.Index of cached texts, memory-mapped embedding matrix)`` for one encoder."""
    if key not in ENCODERS:
        raise ValueError(f"unknown encoder {key!r} (known: {list(ENCODERS)})")
    d, tpath, epath = _cache_paths(key)
    if not (os.path.exists(tpath) and os.path.exists(epath)):
        raise EncoderCacheMissing(f"no {key} encoder cache at {d} — run `{TOOL_CMD}` first")
    with gzip.open(tpath, "rt", encoding="utf-8") as f:
        known = json.load(f)
    E = np.load(epath, mmap_mode="r")
    if len(known) != E.shape[0]:
        raise EncoderCacheMissing(f"{key} cache is inconsistent ({len(known):,} texts vs {E.shape[0]:,} rows) "
                                  f"— delete {d} and re-run `{TOOL_CMD}`")
    return pd.Index(known), E


def _positions(key: str, texts: Sequence[str]):
    idx, E = _open(key)
    pos = idx.get_indexer(list(texts))
    if (pos < 0).any():
        miss = sorted({t for t, p in zip(texts, pos) if p < 0})
        raise EncoderCacheMissing(
            f"{len(miss):,} of {len(set(texts)):,} texts have no {key} embedding in "
            f"{_cache_paths(key)[0]} — run `{TOOL_CMD}` first (e.g. {miss[0][:80]!r})")
    return pos, E


def _rows(E, pos: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Rows ``pos`` of a cache matrix. float16 caches are renormalised after the upcast (fp16
    rounding moves a unit row's norm by ~1e-3); float32 caches are used exactly as stored — they are
    the encoder's own unit-norm output, and renormalising would move the MiniLM rows off the EDA's."""
    X = np.asarray(E[pos], dtype=dtype)
    if E.dtype == np.float16:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = np.divide(X, n, out=np.zeros_like(X), where=n > 0)
    return X


def cache_status(keys: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per encoder: is its cache on disk, how many texts, which dimension and dtype."""
    rows = []
    for key in (keys or list(ENCODERS)):
        d, tpath, epath = _cache_paths(key)
        row = {"encoder": key, "label": ENCODERS[key], "present": os.path.exists(tpath) and os.path.exists(epath),
               "n_texts": np.nan, "dim": np.nan, "dtype": None, "path": d}
        if row["present"]:
            E = np.load(epath, mmap_mode="r")
            row.update(n_texts=E.shape[0], dim=E.shape[1], dtype=str(E.dtype))
        rows.append(row)
    return pd.DataFrame(rows)


def _nonempty(texts: pd.Series) -> np.ndarray:
    """``pref.embed_candidates`` never embeds an empty / whitespace-only completion (``_embed_texts``
    skips it, so its row is dropped); the same rows are dropped here."""
    return texts.astype(str).str.strip().astype(bool).to_numpy()


def required_texts(cands: pd.DataFrame, pool: Optional[pd.DataFrame] = None) -> List[str]:
    """Every text an encoder cache must hold for this frame: the non-empty candidate completions
    (zero-weight ones included) plus the sentence pool (built from the frame unless given)."""
    comp = cands["completion"].astype(str)
    texts = set(comp[_nonempty(comp)])
    if pool is None:
        pool = sentence_pool(comp)
    texts |= set(pool["sentence"])
    return sorted(texts)


def require_caches(cands: pd.DataFrame, keys: Optional[Sequence[str]] = None,
                   pool: Optional[pd.DataFrame] = None) -> None:
    """Raise :class:`EncoderCacheMissing` (naming every short encoder and the tool) unless every
    encoder's cache holds every text :func:`required_texts` asks for. Checks all encoders first, so
    a notebook can skip the whole multi-encoder section rather than export a partial set."""
    texts = required_texts(cands, pool)
    problems = []
    for key in (keys or list(ENCODERS)):
        try:
            _positions(key, texts)
        except EncoderCacheMissing as e:
            problems.append(f"{key}: {str(e).split(' — ')[0]}")
    if problems:
        raise EncoderCacheMissing("; ".join(problems) + f" — run `{TOOL_CMD}`")


def load_encoder_embeddings(texts: Sequence[str], key: str) -> np.ndarray:
    """Unit-norm float32 embeddings of ``texts`` (rows aligned) under encoder ``key``, from its cache.

    Raises :class:`EncoderCacheMissing` naming ``tools/embed_encoders.py`` if any text is not cached
    — this module never loads a model.
    """
    texts = [str(t) for t in texts]
    pos, E = _positions(key, texts)
    return _rows(E, pos)


def embedded_for(cands: pd.DataFrame, key: str) -> pd.DataFrame:
    """``cands`` with an ``emb`` column under encoder ``key`` — the shape
    :func:`pref.embed_candidates` returns (empty completions dropped), so
    :func:`pref.k_direction_cosines_by_iter` / :func:`pref.direction_stability_by_iter` run unchanged.

    Memory: one float32 row per candidate (~2 GB for Llama's 2,048 dimensions over ~239k GRPO
    candidates); :func:`conversation_sums` is the light form the readouts here use.
    """
    if cands.empty:
        return cands
    comp = cands["completion"].astype(str)
    keep = _nonempty(comp)
    out = cands[keep].copy()
    out["emb"] = list(load_encoder_embeddings(comp[keep].tolist(), key))
    return out


def conversation_sums(cands: pd.DataFrame, key: str, *, chunk: int = 16384) -> pd.DataFrame:
    """One row per (arm, train_iter, conversation_id): ``emb`` = Σ w · emb over its candidates
    (float64), ``weight`` = 1.0, plus ``n_candidates``.

    The update direction is linear in its rows, so the pref functions return the same numbers on this
    frame as on :func:`embedded_for`'s — ``_direction`` sums the same float64 products, only in a
    different order (differences ~1e-15) — while touching 96 rows per cell instead of ~12,000.
    Conversation split-halves see the same conversation ids, hence the same seeded draws. Zero-weight
    candidates (which add exactly 0) and empty completions (which ``pref.embed_candidates`` drops)
    are skipped.
    """
    d = cands[(cands["weight"] != 0)]
    comp = d["completion"].astype(str)
    keep = _nonempty(comp)
    d, comp = d[keep], comp[keep]
    if d.empty:
        return pd.DataFrame(columns=_CONV + ["n_candidates", "weight", "emb"])
    from scipy import sparse
    grp = d.groupby(_CONV, sort=True)
    codes = grp.ngroup().to_numpy()                 # numbered in the same sorted order as grp.size()
    out = grp.size().reset_index(name="n_candidates")
    pos, E = _positions(key, comp.tolist())
    w = d["weight"].to_numpy(dtype=np.float64)
    S = np.zeros((len(out), E.shape[1]))
    for s in range(0, len(d), chunk):
        sl = slice(s, s + chunk)
        X = _rows(E, pos[sl], dtype=np.float64)
        M = sparse.csr_matrix((w[sl], (codes[sl], np.arange(X.shape[0]))), shape=(len(out), X.shape[0]))
        S += M @ X
    out["weight"] = 1.0
    out["emb"] = list(S)
    return out


# ── 1 · cosines per encoder ───────────────────────────────────────────────────
def encoder_cosines(cands: pd.DataFrame, keys: Optional[Sequence[str]] = None, *,
                    arm_a: str = _ARM_A, arm_b: str = _ARM_B, n_splits: int = 50,
                    seed: int = BOOT_SEED) -> pd.DataFrame:
    """The paper's two direction tables, once per encoder, in one tidy frame.

    ``kind = "k0_vs_k5"``: :func:`pref.k_direction_cosines_by_iter` (``arm`` = ``"<arm_a> vs
    <arm_b>"``, ``rel_a`` belongs to ``arm_a``); ``kind = "stability"``:
    :func:`pref.direction_stability_by_iter` per run (``train_iter`` = the later iteration,
    ``from_iter`` = the earlier, ``rel_a`` belongs to ``from_iter``). Same estimator, splits and seed
    as the MiniLM tables, so the ``minilm`` rows equal ``direction_k_by_iter_grpo`` /
    ``direction_stability_grpo``. Pass every candidate (zero-weight rows are ignored).
    """
    out = []
    for key in (keys or list(ENCODERS)):
        sums = conversation_sums(cands[cands["arm"].isin([arm_a, arm_b])], key)
        k = _pref.k_direction_cosines_by_iter(sums, arm_a, arm_b, n_splits=n_splits, seed=seed)
        k = k.drop(columns=["arm_a", "arm_b"]).assign(encoder=key, kind="k0_vs_k5", arm=f"{arm_a} vs {arm_b}")
        s = _pref.direction_stability_by_iter(sums, n_splits=n_splits, seed=seed)
        s = s.rename(columns={"to_iter": "train_iter"}).assign(encoder=key, kind="stability")
        out += [k, s]
    cols = ["encoder", "kind", "arm", "from_iter", "train_iter", "raw", "half_cross", "rel_a", "rel_b",
            "corrected", "n_splits"]
    res = pd.concat(out, ignore_index=True).reindex(columns=cols)
    res["from_iter"] = res["from_iter"].astype("Int64")
    return res


# ── 2 · sentence pool + categories ────────────────────────────────────────────
def sentence_pool(texts: Iterable[str], *, min_words: int = 4, max_words: int = 10, min_count: int = 3,
                  max_size: int = 4488) -> pd.DataFrame:
    """Doron's sentence pool: sentences of ``min_words``–``max_words`` words (split on ``.!?`` and
    newlines) seen at least ``min_count`` times verbatim across ``texts`` (every occurrence counts),
    the ``max_size`` most frequent, then every sentence holding a chat marker dropped
    (:data:`POOL_MARKER`: the meetings' :data:`LEAK` plus any ``|im`` — LEAK misses the two marker
    fragments ``What do you think?</|im_end|>`` and ``What do you think?<|im``, which the meetings pool
    kept, so this pool is two sentences smaller than its 4,047).

    Columns ``sentence``, ``count``. Ties at the size cap break by the sentence text (the meetings
    script broke them by first occurrence; on the GRPO candidates the cap does not bind). Sizes are in
    ``.attrs``: ``n_distinct`` (distinct sentences of eligible length), ``n_eligible`` (seen
    ≥ ``min_count`` times), ``n_pool`` (after the cap), ``n_marker`` (dropped for a chat marker).
    """
    c = collections.Counter()
    for t in texts:
        for s in _SPLIT.split(str(t)):
            s = s.strip()
            if min_words <= len(s.split()) <= max_words:
                c[s] += 1
    eligible = sorted(((s, n) for s, n in c.items() if n >= min_count), key=lambda x: (-x[1], x[0]))
    pool = pd.DataFrame(eligible[:max_size], columns=["sentence", "count"])
    marker = pool["sentence"].map(lambda s: bool(POOL_MARKER.search(s))).to_numpy(dtype=bool)
    out = pool[~marker].reset_index(drop=True)
    out.attrs.update(n_distinct=len(c), n_eligible=len(eligible), n_pool=len(pool), n_marker=int(marker.sum()))
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _zs(Ep: np.ndarray, W: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """(n_categories, m) category z on each of the m unit directions (columns of W)."""
    p = Ep @ W
    sd = p.std(0)
    return np.stack([(p[m].mean(0) - p[~m].mean(0)) / sd for m in masks])


def category_scores(cands: pd.DataFrame, keys: Optional[Sequence[str]] = None, *,
                    pool: Optional[pd.DataFrame] = None, categories: Optional[dict] = None,
                    arm_a: str = _ARM_A, arm_b: str = _ARM_B, diff_label: str = "K0-K5",
                    n_boot: int = 500, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Where each sentence category sits along each cell's update direction, per encoder.

    Per encoder × (``arm_a``, ``arm_b``) × train_iter × category: ``z`` = (mean projection of the
    category's pool sentences − mean projection of the other pool sentences) / SD of all the pool's
    projections, on the cell's unit direction (positive = the update pushes toward that kind of
    sentence). ``ci_lo`` / ``ci_hi``: 2.5 / 97.5 percentiles over ``n_boot`` conversation bootstraps
    (each resamples the conversation slots of the cell; conversation_id is the persona slot both runs
    share at an iteration, and the SAME multiplicities are used for every cell and encoder, so the
    ``diff_label`` rows — ``arm_a`` minus ``arm_b`` — are paired). Pool = :func:`sentence_pool` of
    the frame's completions unless given; categories = :data:`CATEGORIES`.
    """
    categories = categories or CATEGORIES
    sub = cands[cands["arm"].isin([arm_a, arm_b])]
    if pool is None:
        pool = sentence_pool(sub["completion"].astype(str))
    sents = pool["sentence"].tolist()
    names = list(categories)
    masks = [np.array([bool(categories[c].search(s)) for s in sents]) for c in names]
    sizes = {c: int(m.sum()) for c, m in zip(names, masks)}
    ids = np.array(sorted(sub["conversation_id"].unique()))
    slot = {v: i for i, v in enumerate(ids)}
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(ids)))
    for b in range(n_boot):
        np.add.at(boot[b], rng.integers(0, len(ids), size=len(ids)), 1.0)

    rows = []
    for key in (keys or list(ENCODERS)):
        sums = conversation_sums(sub, key)
        Ep = load_encoder_embeddings(sents, key).astype(np.float64)
        its = sorted(set(sums.loc[sums["arm"] == arm_a, "train_iter"]) & set(sums.loc[sums["arm"] == arm_b, "train_iter"]))
        for it in its:
            per = {}
            for arm in (arm_a, arm_b):
                g = sums[(sums["arm"] == arm) & (sums["train_iter"] == it)]
                S = np.zeros((len(ids), Ep.shape[1]))
                S[[slot[c] for c in g["conversation_id"]]] = np.vstack(list(g["emb"]))
                per[arm] = (_zs(Ep, _unit(S.sum(0))[:, None], masks)[:, 0], _zs(Ep, _unit(boot @ S).T, masks))
            sides = {arm_a: per[arm_a], arm_b: per[arm_b],
                     diff_label: (per[arm_a][0] - per[arm_b][0], per[arm_a][1] - per[arm_b][1])}
            for side, (pt, bs) in sides.items():
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=1)
                for j, c in enumerate(names):
                    rows.append({"encoder": key, "arm": side, "train_iter": int(it), "category": c,
                                 "n_sentences": sizes[c], "z": float(pt[j]), "ci_lo": float(lo[j]),
                                 "ci_hi": float(hi[j])})
    out = pd.DataFrame(rows)
    out.attrs.update(n_pool=len(sents), n_boot=n_boot, seed=seed)
    return out


def top_sentences(cands: pd.DataFrame, keys: Optional[Sequence[str]] = None, *,
                  pool: Optional[pd.DataFrame] = None, n: int = 5, arm_a: str = _ARM_A, arm_b: str = _ARM_B,
                  diff_label: str = "K0-K5") -> pd.DataFrame:
    """The pool sentences at the two ends of each update direction, per encoder.

    For every encoder × direction × train_iter: the ``n`` pool sentences that project highest
    (``side = "high"``) and lowest (``"low"``) on the cell's unit direction — the same advantage-weighted,
    all-candidates direction :func:`category_scores` scores — and, for ``direction = diff_label``, on
    ``arm_a``'s unit direction minus ``arm_b``'s (``"more K0"`` = highest, ``"more K5"`` = lowest; the
    meetings script's K0 − K5 readout). ``rank`` 1 = most extreme. Long format: ``encoder``,
    ``direction``, ``train_iter``, ``side``, ``rank``, ``sentence``. Pool = :func:`sentence_pool` of the
    frame's completions unless given. Single sentences are illustrations of a direction, not evidence on
    their own: which ones surface shifts between encoders (compare across ``encoder``).
    """
    sub = cands[cands["arm"].isin([arm_a, arm_b])]
    if pool is None:
        pool = sentence_pool(sub["completion"].astype(str))
    sents = np.asarray(pool["sentence"].tolist(), dtype=object)
    rows = []
    for key in (keys or list(ENCODERS)):
        sums = conversation_sums(sub, key)
        Ep = load_encoder_embeddings(sents.tolist(), key).astype(np.float64)
        full = {(a, int(i)): _unit(np.vstack(list(g["emb"])).sum(0))
                for (a, i), g in sums.groupby(["arm", "train_iter"])}
        its = sorted({i for a, i in full if a == arm_a} & {i for a, i in full if a == arm_b})
        for it in its:
            dirs = [(arm_a, full[(arm_a, it)], ("high", "low")), (arm_b, full[(arm_b, it)], ("high", "low")),
                    (diff_label, full[(arm_a, it)] - full[(arm_b, it)], ("more K0", "more K5"))]
            for name, v, (hi, lo) in dirs:
                order = np.argsort(-(Ep @ v), kind="stable")
                for side, idx in ((hi, order[:n]), (lo, order[::-1][:n])):
                    rows += [{"encoder": key, "direction": name, "train_iter": it, "side": side, "rank": r,
                              "sentence": sents[j]} for r, j in enumerate(idx, start=1)]
    return pd.DataFrame(rows)


def category_summary(cat: pd.DataFrame) -> pd.DataFrame:
    """Across encoders, per (arm, train_iter, category): ``z_mean`` / ``z_min`` / ``z_max`` and how
    many encoders' 95% interval lies wholly above (``n_ci_above_0``) or below (``n_ci_below_0``) 0."""
    g = cat.groupby(["arm", "train_iter", "category"], sort=False)
    out = g["z"].agg(z_mean="mean", z_min="min", z_max="max", n_encoders="size").reset_index()
    out["n_ci_above_0"] = g["ci_lo"].apply(lambda s: int((s > 0).sum())).to_numpy()
    out["n_ci_below_0"] = g["ci_hi"].apply(lambda s: int((s < 0).sum())).to_numpy()
    return out


# ── 3 · words, embedding-free ─────────────────────────────────────────────────
def lexical_logodds(cands: pd.DataFrame, *, prior_total: float = 1000.0, min_count: int = 20,
                    top_n: int = 10, pooled: Sequence = (("1-3", 1, 3), ("4-10", 4, 10))) -> pd.DataFrame:
    """Which words win within a group, with no embedder in the loop.

    Monroe, Colaresi & Quinn (2008) log-odds ratio with an informative Dirichlet prior, best vs worst
    candidate of every gradient group: per cell, word counts over the group's top-scoring
    candidate(s) against its bottom-scoring candidate(s), each tied candidate weighted 1/n of its
    tie so every group contributes one unit per side. Prior = ``prior_total`` pseudo-counts spread in
    proportion to each word's frequency over ALL the frame's candidates; vocabulary = lower-case word
    tokens seen at least ``min_count`` times. ``z`` = the log-odds difference over its approximate SD
    (positive = more frequent in the winners).

    Cells: every (arm, train_iter) plus each arm's ``pooled`` iteration ranges (``iterations`` is a
    string column: ``"7"``, ``"4-10"``). Returns the ``top_n`` words per side (``side`` = win /
    lose, ``rank`` 1 = most extreme) with their weighted ``best_count`` / ``worst_count`` (the meetings
    script kept 15; 10 keeps the exported markdown table whole). Pass every candidate
    (``load_weighted_candidates(..., drop_zero_weight=False)``); groups without a reward spread are
    skipped. With every candidate in, fragments of malformed chat markers (``im``, ``end`` from
    ``<|im_end>``) and of degenerate text (single letters) are ordinary tokens and can rank.
    """
    from scipy import sparse
    if cands.empty:
        return pd.DataFrame()
    d = cands.reset_index(drop=True)
    g = d.groupby(_pref._GROUP_KEYS)["score"]
    hi, lo = g.transform("max").to_numpy(), g.transform("min").to_numpy()
    sc = d["score"].to_numpy()
    valid = hi > lo
    best, worst = valid & (sc == hi), valid & (sc == lo)
    gid = d.groupby(_pref._GROUP_KEYS).ngroup().to_numpy()
    nb = np.bincount(gid[best], minlength=gid.max() + 1)
    nw = np.bincount(gid[worst], minlength=gid.max() + 1)
    wb = np.where(best, 1.0 / np.maximum(nb[gid], 1), 0.0)
    ww = np.where(worst, 1.0 / np.maximum(nw[gid], 1), 0.0)

    toks = [_WORDTOK.findall(t.lower()) for t in d["completion"].astype(str)]
    bg = collections.Counter()
    for ts in toks:
        bg.update(ts)
    vocab = sorted(w for w, c in bg.items() if c >= min_count)
    vid = {w: j for j, w in enumerate(vocab)}
    r, c = [], []
    for i, ts in enumerate(toks):
        for t in ts:
            j = vid.get(t)
            if j is not None:
                r.append(i)
                c.append(j)
    X = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(d), len(vocab)))
    bgv = np.array([bg[w] for w in vocab], dtype=float)
    alpha = prior_total * bgv / bgv.sum()
    a0 = alpha.sum()

    arm, it = d["arm"].to_numpy(), d["train_iter"].to_numpy()
    cells = [(a, str(int(i)), (arm == a) & (it == i)) for a in sorted(set(arm)) for i in sorted(set(it[arm == a]))]
    cells += [(a, lab, (arm == a) & (it >= i0) & (it <= i1)) for a in sorted(set(arm)) for lab, i0, i1 in pooled]
    rows = []
    for a, lab, m in cells:
        if not m.any():
            continue
        yb, yw = X.T @ (wb * m), X.T @ (ww * m)
        n1, n2 = yb.sum(), yw.sum()
        delta = (np.log((yb + alpha) / (n1 + a0 - yb - alpha)) - np.log((yw + alpha) / (n2 + a0 - yw - alpha)))
        z = delta / np.sqrt(1.0 / (yb + alpha) + 1.0 / (yw + alpha))
        n_groups = int(len(np.unique(gid[m & valid])))
        for side, order in (("win", np.argsort(-z, kind="stable")), ("lose", np.argsort(z, kind="stable"))):
            for rank, j in enumerate(order[:top_n], start=1):
                rows.append({"arm": a, "iterations": lab, "side": side, "rank": rank, "word": vocab[j],
                             "z": float(z[j]), "best_count": float(yb[j]), "worst_count": float(yw[j]),
                             "n_groups": n_groups})
    return pd.DataFrame(rows)


def clean_mask(cands: pd.DataFrame) -> np.ndarray:
    """``True`` for the candidates the exploratory "clean" set keeps: no leaked chat marker (:data:`LEAK`) AND not
    degenerate text (:func:`degenerate_text`, computed over ``cands`` — pass every candidate of both GRPO runs)."""
    texts = cands["completion"].astype(str).tolist()
    leak = np.array([bool(LEAK.search(t)) for t in texts])
    degen = degenerate_text(texts).astype(bool)
    return ~leak & ~degen


def clean_counts(cands: pd.DataFrame) -> pd.DataFrame:
    """Per arm (plus an ``all`` row): what the clean filter drops — ``n_leak`` (leaked chat marker), ``n_degenerate``
    (degenerate text), ``n_dropped`` (either; a candidate can be both), ``share_dropped``, and the gradient groups
    before (``n_groups``) and after (``n_groups_kept`` = groups whose kept candidates still hold two rewards)."""
    texts = cands["completion"].astype(str).tolist()
    d = cands[["arm"] + _pref._GROUP_KEYS[1:] + ["score"]].copy()
    d["leak"] = np.array([bool(LEAK.search(t)) for t in texts])
    d["degen"] = degenerate_text(texts).astype(bool)
    d["drop"] = d["leak"] | d["degen"]

    def row(sub: pd.DataFrame, arm) -> dict:
        kept = sub[~sub["drop"]]
        spread = kept.groupby(_pref._GROUP_KEYS)["score"].agg(lambda s: s.max() > s.min())
        return {"arm": arm, "n_candidates": len(sub), "n_leak": int(sub["leak"].sum()),
                "n_degenerate": int(sub["degen"].sum()), "n_dropped": int(sub["drop"].sum()),
                "share_dropped": float(sub["drop"].mean()), "n_groups": int(sub.groupby(_pref._GROUP_KEYS).ngroups),
                "n_groups_kept": int(spread.sum())}

    return pd.DataFrame([row(g, a) for a, g in d.groupby("arm")] + [row(d, "all")])


def lexical_logodds_clean(cands: pd.DataFrame, **kw) -> pd.DataFrame:
    """:func:`lexical_logodds` on the exploratory "clean" set — the version to quote.

    Drops every candidate with a leaked chat marker or degenerate text (:func:`clean_mask`, the meetings
    definitions copied verbatim), then re-chooses each group's best and worst among the KEPT candidates (a group
    whose kept candidates no longer differ in reward is skipped); the prior and the vocabulary are built from the
    kept candidates too — exactly the meetings script's ``lexical_logodds(D, keeps["clean"])``. On all candidates
    the losing side is dominated by single-letter fragments of degenerate text, and pieces of the malformed
    ``<|im_end>`` marker (``im``, ``end``) rank among K=0's late winners; this removes both. ``kw`` passes through
    (``top_n``, ``prior_total``, ``min_count``, ``pooled``). Pass every candidate of both GRPO runs.
    """
    if cands.empty:
        return pd.DataFrame()
    return lexical_logodds(cands[clean_mask(cands)], **kw)


# ── figure ────────────────────────────────────────────────────────────────────
def plot_encoder_categories(summary: pd.DataFrame, *, categories: Sequence[str] = ("praise", "question", "advice"),
                            arms: Sequence[str] = (_ARM_A, _ARM_B), n_sentences: Optional[dict] = None):
    """One panel per category: each run's ``z`` along its update direction by training iteration —
    the mean over encoders as the line, the min–max over encoders as the band (from
    :func:`category_summary`). K=0 / K=5 drawn in the arm palette with the shared K line style."""
    import matplotlib.pyplot as plt
    from .plotting_style import arm_palette
    from .plotting._shared import K_STYLE
    pal = arm_palette(list(arms))
    fig, axes = plt.subplots(1, len(categories), figsize=(3.6 * len(categories), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cat in zip(axes, categories):
        for arm in arms:
            g = summary[(summary["arm"] == arm) & (summary["category"] == cat)].sort_values("train_iter")
            st = K_STYLE.get(k_of(arm), K_STYLE[0])
            ax.fill_between(g["train_iter"], g["z_min"], g["z_max"], color=pal[arm], alpha=0.18, lw=0)
            ax.plot(g["train_iter"], g["z_mean"], color=pal[arm], ls=st["ls"], marker=st["marker"], ms=4, lw=1.5,
                    label=f"{arm} (K={k_of(arm)})")
        ax.axhline(0, color="#444444", lw=0.8)
        ax.grid(True, alpha=0.3)
        n = (n_sentences or {}).get(cat)
        ax.set_title(f"{cat}" + (f"  ({n} pool sentences)" if n else ""), fontsize=9)
        ax.set_xticks(range(1, int(summary["train_iter"].max()) + 1))
        ax.set_xlabel("training iteration (train_iter)", fontsize=8)
    axes[0].set_ylabel("z: category vs rest of the pool\n(pool SDs along the update direction)", fontsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    return fig
