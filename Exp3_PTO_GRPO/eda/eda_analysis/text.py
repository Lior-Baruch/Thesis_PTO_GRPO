"""
text.py — judge-free text + embedding evals on the EVAL conversations (family ``lookahead/text``).

Every artifact in the results tree that touches the transcripts is a per-conversation RATE
(``behavior.text_metrics``: length, ``?`` per turn, loops, two lexical markers). Nothing looks at
the utterances themselves as *content*. This module does, in the sentence-embedding space the
training-side probe already uses (``pref.py``, ``all-MiniLM-L6-v2``), so the eval side and the
update side are comparable:

  A1  **Repertoire** — cluster the BASE policy's therapist utterances into a repertoire of
      behaviours; track cluster OCCUPANCY per arm × iteration (+ an out-of-repertoire ``novel``
      share). Clusters that grow are *learned*, clusters that shrink are *unlearned*.
  A2  **Drift** — centroid displacement from base per state, the cosine between the two K arms'
      displacements (did they learn the same thing?), and the alignment between what the update
      selected for (``pref.direction_by_arm``) and what the policy actually became.
  A3  **Diversity / persona sensitivity** — template similarity across personas at matched turn
      index, within-conversation self-similarity, a between-persona variance share (does the
      therapist tailor to the patient?), distinct-n and a cross-conversation near-duplicate rate.
  A4  **Semantic echo** — cosine of a therapist turn to the patient turn it answers, minus a
      shuffled baseline: a judge-free *reflection* proxy, cross-checked against the oracle's MITI
      reflection counts (the way ``?``-rate is cross-checked against ``MITI_B3_Q``).
  A5  **Within-session profile** — every per-turn feature by turn index (early / mid / late).
  A6  **Patient side** — patient turn length, questions, a disengagement marker, and the
      patient's echo of the therapist.

Conventions shared with the rest of the package: frames are tidy, nothing here writes to disk
(the notebook exports), bootstraps seed with :data:`constants.BOOT_SEED`, per-conversation
metrics are persona-paired through :func:`behavior._attach_by_arm`, and the scripted therapist
opener (utterance 0 of every conversation, identical text) is excluded from every content
statistic — it is the prompt, not the policy.

Embeddings: :func:`pref._embed_texts` (sha1-keyed pickle cache) pointed at its own cache dir
(``eda/.emb_cache/eval/``) so the eval-side cache never rewrites the 300 MB training-side one.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import BOOT_SEED, RE_AFFIRM, RE_EFFUSIVE, PERSONA_COLS
from .data import load_cached, conv_input_roots
from . import pref as _pref

_STATE = ["arm", "method", "K", "model", "iteration", "is_base"]
_CONV = _STATE + ["file_index"]
_EVAL_CACHE_DIR = os.path.join(_pref._CACHE_DIR, "eval")
DEFAULT_MODEL = _pref._DEFAULT_MODEL

# Patient-side disengagement cues (case-insensitive). Like ``RE_AFFIRM``/``RE_EFFUSIVE`` these
# are DIRECTIONAL sanity markers, not a measurement — their absolute level is meaningless.
RE_DISENGAGE = re.compile(
    r"\b(i don'?t know|not sure|i guess|whatever|i suppose|don'?t (?:really )?(?:want|care|see the point)|"
    r"told to (?:be|come)|leave me|waste of time|not interested|doesn'?t matter|i'?m fine)\b", re.I)

_TOK = re.compile(r"[a-z']+")
_STOP = set("""a an the and or but if then so of to in on at for with from by as is are was were be
been being am it its it's this that these those i you he she we they me him her us them my your his
our their mine yours what which who whom whose when where why how all any both each few more most
other some such no nor not only own same than too very can will just do does did doing have has had
having would could should may might must shall about above after again against between into through
during before under over out up down off further here there once s t don ve ll re d m""".split())

# Turn-index bins over the therapist's OWN turn counter (opener = 0, excluded).
PROFILE_BINS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 10 ** 6, "10+")]

#: Per-conversation metrics that get the persona-paired K contrast (through ``to_scores_long``).
TEXT_K_METRICS = ["dist_to_base", "novel_share", "within_sim", "echo", "lex_recall_prev",
                  "patient_echo", "pt_turn_len", "pt_q_per_turn", "pt_disengage_rate"]
TEXT_METRIC_LABELS = {
    "dist_to_base": "distance to base (1 − cos of the conversation's therapist centroid to the pooled base centroid)",
    "novel_share": "out-of-repertoire share of therapist turns",
    "within_sim": "within-conversation therapist self-similarity (mean pairwise cos)",
    "echo": "responsiveness: semantic echo of the preceding patient turn (cos, baseline-subtracted)",
    "lex_recall_prev": "responsiveness: share of the preceding patient turn's content words re-used by the therapist",
    "patient_echo": "patient responsiveness: echo of the preceding therapist turn (cos, baseline-subtracted)",
    "pt_turn_len": "patient turn length (chars)",
    "pt_q_per_turn": "'?' per patient turn",
    "pt_disengage_rate": "share of patient turns with a disengagement cue",
}
LOWER_BETTER = {"pt_disengage_rate", "within_sim"}
#: Therapist-side per-turn features that get the within-session profile.
PROFILE_FEATURES = ["n_chars", "q_count", "effusive", "affirm", "echo", "lex_recall_prev"]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Utterances + embeddings                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _arms(arms):
    from . import discover_arms
    return discover_arms() if arms is None else arms


def load_utterances(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """One row per utterance of every eval conversation of every arm × iteration.

    Columns: the conversation keys (``arm, method, K, model, iteration, is_base, file_index``),
    ``utt_idx`` (position in the conversation), ``role``, ``role_turn`` (index among that role's
    turns — the therapist opener is ``role_turn == 0``), ``is_opener``, ``text``, ``n_chars``,
    ``n_words``, ``q_count``, ``effusive``, ``affirm``, ``disengage`` (+ persona columns).
    Parquet-cached on the conversation CSVs.
    """
    arms = _arms(arms)
    return load_cached("utterances", arms,
                       lambda: _load_utterances_impl(arms, attach_persona=attach_persona),
                       input_roots=conv_input_roots(arms),
                       params={"attach_persona": attach_persona, "v": 1})


def _load_utterances_impl(arms, *, attach_persona: bool) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for k in arm.iters:
            cdir = arm.conv_dir(k)
            if not cdir or not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                m = re.match(r"conversation_(\d+)\.csv$", fn)
                if not m:
                    continue
                try:
                    cdf = pd.read_csv(os.path.join(cdir, fn))
                except Exception:
                    continue
                if "role" not in cdf.columns or "conversation" not in cdf.columns:
                    continue
                counters = {"therapist": 0, "patient": 0}
                for i, (role, text) in enumerate(zip(cdf["role"].astype(str), cdf["conversation"])):
                    text = "" if pd.isna(text) else str(text)
                    rt = counters.get(role, 0)
                    counters[role] = rt + 1
                    rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                                 "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                                 "file_index": int(m.group(1)), "utt_idx": i, "role": role,
                                 "role_turn": rt, "is_opener": (role == "therapist" and i == 0),
                                 "text": text, "n_chars": len(text),
                                 "n_words": len(_TOK.findall(text.lower())),
                                 "q_count": text.count("?"),
                                 "effusive": bool(RE_EFFUSIVE.search(text)),
                                 "affirm": bool(RE_AFFIRM.search(text)),
                                 "disengage": bool(RE_DISENGAGE.search(text))})
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        from .behavior import _attach_by_arm
        df = _attach_by_arm(df, arms)
    return df


def embed_utterances(utt: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Unit-norm embedding matrix aligned to ``utt``'s rows (zeros for empty text)."""
    texts = utt["text"].astype(str).tolist()
    lut = _pref._embed_texts(texts, model_name, cache_dir=_EVAL_CACHE_DIR)
    dim = next(iter(lut.values())).shape[0] if lut else 384
    E = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        v = lut.get(t)
        if v is not None:
            E[i] = v
    return E


def content_mask(utt: pd.DataFrame, role: str = "therapist") -> np.ndarray:
    """Rows that count as policy CONTENT: the role's turns minus the scripted opener + empties."""
    _check_aligned(utt)
    return ((utt["role"] == role) & ~utt["is_opener"] & (utt["n_chars"] > 0)).to_numpy()


def _check_aligned(utt: pd.DataFrame) -> None:
    """Every function here indexes the embedding matrix by ROW POSITION, so the utterance frame
    must carry a clean RangeIndex (``utt.reset_index(drop=True)`` after any filtering)."""
    if not (isinstance(utt.index, pd.RangeIndex) and utt.index.start == 0 and utt.index.step == 1):
        raise ValueError("utterance frame must have a 0..n-1 RangeIndex aligned with the embedding "
                         "matrix — call utt.reset_index(drop=True) and re-embed after filtering")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  A1 · Repertoire — cluster the base policy's behaviours, track occupancy       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def repertoire_fit(utt: pd.DataFrame, E: np.ndarray, *, k: int = 30, seed: int = BOOT_SEED,
                   novel_pct: float = 5.0) -> dict:
    """Fit the base repertoire: k-means on the BASE policy's therapist utterances.

    The base pool is every arm's ``iteration == 0`` (all four arms start from the same policy, so
    pooling them is four independent draws of the same repertoire). Personas are split in half:
    even ``persona_id`` fits the centroids, odd calibrates the *novelty threshold* (the
    ``novel_pct``-th percentile of held-out nearest-centroid similarity), so the base's own novel
    share is out-of-sample rather than 5 % by construction. Returns a dict with ``centers`` (unit
    rows), ``threshold``, ``k``, ``seed``, ``fit_idx`` / ``calib_idx`` (row indices into ``utt``).
    """
    from sklearn.cluster import KMeans
    base = content_mask(utt) & (utt["iteration"] == 0).to_numpy()
    pid = utt["persona_id"].to_numpy() if "persona_id" in utt.columns else utt["file_index"].to_numpy()
    fit_idx = np.flatnonzero(base & (pid % 2 == 0))
    calib_idx = np.flatnonzero(base & (pid % 2 == 1))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(E[fit_idx])
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    sims = (E[calib_idx] @ centers.T).max(axis=1)
    return {"centers": centers, "threshold": float(np.percentile(sims, novel_pct)), "k": int(k),
            "seed": int(seed), "novel_pct": novel_pct, "fit_idx": fit_idx, "calib_idx": calib_idx,
            "n_fit": int(fit_idx.size), "n_calib": int(calib_idx.size)}


def repertoire_assign(utt: pd.DataFrame, E: np.ndarray, rep: dict) -> pd.DataFrame:
    """Every therapist content utterance → nearest base cluster, its similarity, ``novel`` flag."""
    mask = content_mask(utt)
    idx = np.flatnonzero(mask)
    S = E[idx] @ rep["centers"].T
    out = utt.loc[idx, _CONV + ["utt_idx", "role_turn", "text", "n_chars", "q_count",
                              "effusive", "affirm"]].copy()
    out["cluster"] = S.argmax(axis=1)
    out["sim"] = S.max(axis=1)
    out["novel"] = out["sim"] < rep["threshold"]
    return out.reset_index(drop=True)


def repertoire_occupancy(assigned: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per conversation: share of therapist turns in each cluster (``c0..c{k-1}``) + ``novel_share``.

    Novel turns keep their nearest-cluster assignment for the ``c*`` shares (so shares sum to 1)
    and are additionally counted in ``novel_share``.
    """
    rows = []
    for key, g in assigned.groupby(_CONV, sort=False):
        n = len(g)
        counts = np.bincount(g["cluster"].to_numpy(), minlength=k) / max(n, 1)
        row = dict(zip(_CONV, key))
        row.update({f"c{i}": float(counts[i]) for i in range(k)})
        row["novel_share"] = float(g["novel"].mean())
        row["n_turns"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def repertoire_by_state(occ: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per (arm, iteration): mean cluster shares, ``novel_share``, occupancy entropy (nats) and
    the effective number of clusters ``exp(H)``."""
    cols = [f"c{i}" for i in range(k)]
    rows = []
    for key, g in occ.groupby(_STATE, sort=False):
        p = g[cols].mean().to_numpy()
        p = p / p.sum() if p.sum() > 0 else p
        nz = p[p > 0]
        H = float(-(nz * np.log(nz)).sum())
        row = dict(zip(_STATE, key))
        row.update({c: float(v) for c, v in zip(cols, p)})
        row.update({"novel_share": float(g["novel_share"].mean()), "entropy": H,
                    "eff_clusters": float(np.exp(H)), "n_convs": int(len(g)),
                    "n_turns": int(g["n_turns"].sum())})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "iteration"]).reset_index(drop=True)


def _top_words(texts: Sequence[str], ref_counter: Counter, ref_total: int, n: int = 4) -> str:
    # Distinctiveness weighted by in-cluster frequency (p_in · log-odds, the KL contribution), so a
    # word that is both common in the cluster and rare elsewhere wins over a rare-everywhere word
    # that happens to sit in this cluster three times. Also requires presence in ≥ 10 % of turns.
    c = Counter(w for t in texts for w in _TOK.findall(t.lower()) if w not in _STOP and len(w) > 2)
    doc = Counter(w for t in texts for w in set(_TOK.findall(t.lower())) if w not in _STOP and len(w) > 2)
    tot = sum(c.values()) or 1
    n_docs = max(len(texts), 1)
    scored = []
    for w, f in c.items():
        if f < 3 or doc.get(w, 0) < 0.10 * n_docs:
            continue
        p_in = (f + 0.5) / (tot + 1)
        p_out = (ref_counter.get(w, 0) - f + 0.5) / (max(ref_total - tot, 1) + 1)
        scored.append((p_in * np.log(p_in / p_out), w))
    return " ".join(w for _, w in sorted(scored, reverse=True)[:n])


def repertoire_labels(utt: pd.DataFrame, E: np.ndarray, assigned: pd.DataFrame, rep: dict,
                      *, n_exemplars: int = 3, width: int = 140) -> pd.DataFrame:
    """Human-readable cluster cards: distinctive words, lexical tags, MI word-category hits, and
    the ``n_exemplars`` base utterances nearest the centroid."""
    base = assigned[assigned["iteration"] == 0]
    ref = Counter(w for t in base["text"] for w in _TOK.findall(t.lower())
                  if w not in _STOP and len(w) > 2)
    ref_total = sum(ref.values())
    cats = _pref.MI_CATEGORIES
    rows = []
    for c in range(rep["k"]):
        g = base[base["cluster"] == c]
        texts = g["text"].tolist()
        hits = {}
        for cat, words in cats.items():
            pat = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.I)
            hits[f"mi_{cat}"] = float(np.mean([bool(pat.search(t)) for t in texts])) if texts else np.nan
        # Exemplars: nearest base utterances to the centroid.
        ex = ""
        if len(g):
            sims = E[_rows_of(utt, g)] @ rep["centers"][c]
            order = np.argsort(-sims)[:n_exemplars]
            ex = " ‖ ".join(_clip(texts[i], width) for i in order)
        rows.append({"cluster": c, "n_base": int(len(g)),
                     "share_base": float(len(g) / max(len(base), 1)),
                     "top_words": _top_words(texts, ref, ref_total),
                     "q_rate": float(np.mean(g["q_count"] > 0)) if len(g) else np.nan,
                     "effusive_rate": float(g["effusive"].mean()) if len(g) else np.nan,
                     "affirm_rate": float(g["affirm"].mean()) if len(g) else np.nan,
                     "mean_chars": float(g["n_chars"].mean()) if len(g) else np.nan,
                     **hits, "exemplars": ex})
    return pd.DataFrame(rows)


def _rows_of(utt: pd.DataFrame, g: pd.DataFrame) -> np.ndarray:
    """Row positions in ``utt`` of the utterances in ``g`` (matched on conversation + utt_idx)."""
    key = _CONV + ["utt_idx"]
    lut = pd.Series(np.arange(len(utt)), index=pd.MultiIndex.from_frame(utt[key]))
    return lut.loc[pd.MultiIndex.from_frame(g[key])].to_numpy()


def _clip(t: str, width: int) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= width else t[: width - 1] + "…"


def learned_unlearned(state: pd.DataFrame, labels: pd.DataFrame, k: int, *, top: int = 6,
                      final_iter: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Per arm: the ``top`` clusters that grew most and shrank most, final vs the arm's own base.

    ``delta`` is in share-of-therapist-turns; ``rel`` = delta / share_base. A cluster with
    ``share_base`` ≈ 0 that grows is a learned behaviour; one with a large base share that goes to
    ≈ 0 is unlearned. ``final_iter`` overrides the per-arm final iteration (default: last).
    """
    cols = [f"c{i}" for i in range(k)]
    lab = labels.set_index("cluster")
    rows = []
    for arm, g in state.groupby("arm", sort=False):
        g = g.sort_values("iteration")
        b = g[g["iteration"] == 0]
        fi = (final_iter or {}).get(arm, int(g["iteration"].max()))
        f = g[g["iteration"] == fi]
        if b.empty or f.empty:
            continue
        d = f[cols].iloc[0].to_numpy() - b[cols].iloc[0].to_numpy()
        order = np.argsort(d)
        picks = [(int(c), "learned") for c in order[::-1][:top] if d[c] > 0] + \
                [(int(c), "unlearned") for c in order[:top] if d[c] < 0]
        for c, direction in picks:
            rows.append({"arm": arm, "final_iter": fi, "direction": direction, "cluster": c,
                         "share_base": float(b[f"c{c}"].iloc[0]), "share_final": float(f[f"c{c}"].iloc[0]),
                         "delta": float(d[c]),
                         "top_words": lab.loc[c, "top_words"], "q_rate": lab.loc[c, "q_rate"],
                         "effusive_rate": lab.loc[c, "effusive_rate"],
                         "exemplar": lab.loc[c, "exemplars"].split(" ‖ ")[0]})
    return pd.DataFrame(rows)


def repertoire_stability(utt: pd.DataFrame, E: np.ndarray, *, ks: Sequence[int] = (20, 30, 40),
                         seeds: Sequence[int] = (0, 1, 2), ref_k: int = 30,
                         ref_seed: int = BOOT_SEED) -> pd.DataFrame:
    """How much of the repertoire reading depends on ``k`` and the k-means seed.

    For every (k, seed) refit: Spearman ρ of the per-state ``novel_share`` and ``entropy`` series
    against the reference fit, and the adjusted Rand index of the base-pool labels vs the
    reference at the same ``k``. The per-state scalars are the quantities comparable across
    ``k``; cluster identities are not.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    ref = repertoire_fit(utt, E, k=ref_k, seed=ref_seed)
    ref_assigned = repertoire_assign(utt, E, ref)
    ref_state = repertoire_by_state(repertoire_occupancy(ref_assigned, ref_k), ref_k)
    ref_state = ref_state.set_index(["arm", "iteration"])
    ref_base_labels = ref_assigned.loc[ref_assigned["iteration"] == 0, "cluster"].to_numpy()
    rows = []
    for k in ks:
        for s in seeds:
            rep = repertoire_fit(utt, E, k=k, seed=s)
            a = repertoire_assign(utt, E, rep)
            st = repertoire_by_state(repertoire_occupancy(a, k), k).set_index(["arm", "iteration"])
            st = st.reindex(ref_state.index)
            rho_novel = spearmanr(st["novel_share"], ref_state["novel_share"]).correlation
            rho_ent = spearmanr(st["entropy"], ref_state["entropy"]).correlation
            ari = adjusted_rand_score(ref_base_labels,
                                      a.loc[a["iteration"] == 0, "cluster"].to_numpy()) \
                if k == ref_k else np.nan
            rows.append({"k": k, "seed": s, "threshold": rep["threshold"],
                         "rho_novel_share": float(rho_novel), "rho_entropy": float(rho_ent),
                         "ari_base_labels_vs_ref": float(ari) if np.isfinite(ari) else np.nan,
                         "novel_share_base_mean": float(st.loc[st.index.get_level_values(1) == 0,
                                                              "novel_share"].mean()),
                         "novel_share_final_mean": float(st.loc[st.index.get_level_values(1) ==
                                                               st.index.get_level_values(1).max(),
                                                               "novel_share"].mean())})
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  A2 · Drift — where each policy moved, and whether the two K arms moved alike  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def state_centroids(utt: pd.DataFrame, E: np.ndarray, role: str = "therapist") -> Dict[tuple, np.ndarray]:
    """``{(arm, iteration): mean embedding}`` over the role's content utterances."""
    mask = content_mask(utt, role)
    out = {}
    for key, g in utt[mask].groupby(["arm", "iteration"], sort=False):
        out[(key[0], int(key[1]))] = E[g.index.to_numpy()].mean(axis=0)
    return out


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def drift_by_state(centroids: Dict[tuple, np.ndarray]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Per (arm, iteration): displacement from the POOLED base centroid and from the arm's own.

    Columns: ``drift_norm`` (‖c − c_base_pooled‖), ``drift_norm_own`` (vs the arm's iteration 0),
    ``step_norm`` (‖c_iter − c_iter-1‖), ``cos_to_final`` (cos(d_iter, d_final): is the path a
    straight line?). Returns the frame and the pooled base centroid.
    """
    arms = sorted({a for a, _ in centroids})
    base_own = {a: centroids.get((a, 0)) for a in arms}
    pooled = np.mean([v for v in base_own.values() if v is not None], axis=0)
    rows = []
    for a in arms:
        its = sorted(i for aa, i in centroids if aa == a)
        d_final = centroids[(a, its[-1])] - pooled
        prev = None
        for i in its:
            c = centroids[(a, i)]
            d = c - pooled
            rows.append({"arm": a, "iteration": i, "drift_norm": float(np.linalg.norm(d)),
                         "drift_norm_own": float(np.linalg.norm(c - base_own[a]))
                         if base_own[a] is not None else np.nan,
                         "step_norm": float(np.linalg.norm(c - prev)) if prev is not None else np.nan,
                         "cos_to_final": _cos(d, d_final) if i > 0 else np.nan})
            prev = c
    return pd.DataFrame(rows), pooled


def drift_cosines(centroids: Dict[tuple, np.ndarray], pooled: np.ndarray) -> pd.DataFrame:
    """Per iteration: cos between displacement vectors — the two K arms of one method
    (``cos_K0_K5``) and the two methods at one K (``cos_PTO_GRPO``)."""
    def d(arm, it):
        v = centroids.get((arm, it))
        return None if v is None else v - pooled
    its = sorted({i for _, i in centroids if i > 0})
    rows = []
    for it in its:
        row = {"iteration": it}
        for m in ("PTO", "GRPO"):
            a, b = d(f"{m}_LA0", it), d(f"{m}_LA5", it)
            row[f"cos_K0_K5_{m}"] = _cos(a, b) if a is not None and b is not None else np.nan
        for K in (0, 5):
            a, b = d(f"PTO_LA{K}", it), d(f"GRPO_LA{K}", it)
            row[f"cos_PTO_GRPO_K{K}"] = _cos(a, b) if a is not None and b is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def update_alignment(centroids: Dict[tuple, np.ndarray], pooled: np.ndarray, arms) -> pd.DataFrame:
    """cos(update direction, realised drift): does the policy move the way the update pushed?

    ``dir_pooled`` = :func:`pref.direction_by_arm` (the arm's whole-run update direction, in the
    same embedding space); ``dir_iter`` = :func:`pref.direction_by_iter` at ``train_iter = i``,
    which is the update that turned ``model_iter i-1`` into ``model_iter i``. Rows per (arm,
    iteration ≥ 1): ``cos_pooled_vs_cumdrift`` (pooled direction vs c_i − c_base), ``cos_iter_vs_step``
    (that iteration's direction vs c_i − c_{i−1}), ``cos_iter_vs_cumdrift``. Per-iteration
    directions are noisy for PTO (split-half ≈ 0.2) — read those rows next to
    ``arms/preference/tables/*/update_direction_quality.md``.
    """
    cands = _pref.load_weighted_candidates(arms)
    if cands.empty:
        return pd.DataFrame()
    emb = _pref.embed_candidates(cands)
    by_arm = _pref.direction_by_arm(emb)
    by_iter = _pref.direction_by_iter(emb)
    rows = []
    for (a, i), c in sorted(centroids.items()):
        if i == 0 or a not in by_arm:
            continue
        cum = c - pooled
        prev = centroids.get((a, i - 1))
        step = c - prev if prev is not None else None
        di = by_iter.get(a, {}).get(i)
        rows.append({"arm": a, "iteration": i,
                     "cos_pooled_vs_cumdrift": _cos(by_arm[a], cum),
                     "cos_iter_vs_step": _cos(di, step) if di is not None and step is not None else np.nan,
                     "cos_iter_vs_cumdrift": _cos(di, cum) if di is not None else np.nan})
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  A3 · Diversity + A4 · Echo + A6 · Patient side — per conversation and per state ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _mean_pairwise_cos(X: np.ndarray) -> float:
    """Mean cosine over all distinct pairs of the rows of a unit-norm matrix (closed form)."""
    n = X.shape[0]
    if n < 2:
        return np.nan
    s = X.sum(axis=0)
    return float((s @ s - n) / (n * (n - 1)))


def utterance_features(utt: pd.DataFrame, E: np.ndarray, *, n_baseline: int = 32,
                       seed: int = BOOT_SEED) -> Dict[str, np.ndarray]:
    """Per-utterance arrays aligned to ``utt``'s rows (NaN where undefined):

    ``echo`` (therapist turns) = cos(turn, preceding patient turn) − mean cos(turn, ``n_baseline``
    random patient turns from OTHER conversations of the same state); ``patient_echo`` (patient
    turns) mirrors it against the preceding therapist turn (baseline = random non-opener therapist
    turns of the same state); ``lex_recall_prev`` = :func:`lexical_recall_prev`. Baseline draws are
    seeded per state.
    """
    th_mask = content_mask(utt, "therapist")
    pt_mask = content_mask(utt, "patient")
    rng = np.random.default_rng(seed)
    per_utt_echo = np.full(len(utt), np.nan)
    per_utt_pecho = np.full(len(utt), np.nan)
    for key, g in utt.groupby(_STATE, sort=False):
        gi = g.index.to_numpy()
        th_idx = gi[th_mask[gi]]
        pt_idx = gi[pt_mask[gi]]
        if th_idx.size == 0 or pt_idx.size == 0:
            continue
        b_pt = rng.choice(pt_idx, size=min(n_baseline, pt_idx.size), replace=False)
        b_th = rng.choice(th_idx, size=min(n_baseline, th_idx.size), replace=False)
        base_pt = (E[th_idx] @ E[b_pt].T).mean(axis=1)
        base_th = (E[pt_idx] @ E[b_th].T).mean(axis=1)
        pos = {(fi, ui): i for fi, ui, i in zip(g["file_index"], g["utt_idx"], gi)}
        prev_th = np.array([pos.get((fi, ui - 1), -1) for fi, ui in
                            zip(utt.loc[th_idx, "file_index"], utt.loc[th_idx, "utt_idx"])])
        prev_pt = np.array([pos.get((fi, ui - 1), -1) for fi, ui in
                            zip(utt.loc[pt_idx, "file_index"], utt.loc[pt_idx, "utt_idx"])])
        ok = prev_th >= 0
        raw = np.einsum("ij,ij->i", E[th_idx[ok]], E[prev_th[ok]])
        per_utt_echo[th_idx[ok]] = raw - base_pt[ok]
        ok2 = prev_pt >= 0
        raw2 = np.einsum("ij,ij->i", E[pt_idx[ok2]], E[prev_pt[ok2]])
        per_utt_pecho[pt_idx[ok2]] = raw2 - base_th[ok2]
    return {"echo": per_utt_echo, "patient_echo": per_utt_pecho, "lex_recall_prev": lexical_recall_prev(utt)}


def per_conv_metrics(utt: pd.DataFrame, E: np.ndarray, pooled_base: np.ndarray, *,
                     feats: Optional[Dict[str, np.ndarray]] = None, n_baseline: int = 32,
                     seed: int = BOOT_SEED) -> pd.DataFrame:
    """One row per conversation: the per-conversation text metrics (:data:`TEXT_K_METRICS` minus
    ``novel_share``, which comes from :func:`repertoire_occupancy`). ``feats`` =
    :func:`utterance_features` (computed here if not given)."""
    th_mask = content_mask(utt, "therapist")
    pt_mask = content_mask(utt, "patient")
    feats = feats or utterance_features(utt, E, n_baseline=n_baseline, seed=seed)
    per_utt_echo, per_utt_pecho, per_utt_recall = feats["echo"], feats["patient_echo"], feats["lex_recall_prev"]
    rows = []
    for key, g in utt.groupby(_STATE, sort=False):
        for fi, c in g.groupby("file_index", sort=False):
            ci = c.index.to_numpy()
            t = ci[th_mask[ci]]
            p = ci[pt_mask[ci]]
            cen = E[t].mean(axis=0) if t.size else None
            row = dict(zip(_STATE, key))
            row.update({
                "file_index": int(fi),
                "dist_to_base": 1.0 - _cos(cen, pooled_base) if cen is not None else np.nan,
                "within_sim": _mean_pairwise_cos(E[t]) if t.size >= 2 else np.nan,
                "echo": float(np.nanmean(per_utt_echo[t])) if t.size and np.isfinite(per_utt_echo[t]).any() else np.nan,
                "lex_recall_prev": float(np.nanmean(per_utt_recall[t])) if t.size and np.isfinite(per_utt_recall[t]).any() else np.nan,
                "patient_echo": float(np.nanmean(per_utt_pecho[p])) if p.size and np.isfinite(per_utt_pecho[p]).any() else np.nan,
                "pt_turn_len": float(utt.loc[p, "n_chars"].mean()) if p.size else np.nan,
                "pt_q_per_turn": float(utt.loc[p, "q_count"].mean()) if p.size else np.nan,
                "pt_disengage_rate": float(utt.loc[p, "disengage"].mean()) if p.size else np.nan,
                "n_th_turns": int(t.size), "n_pt_turns": int(p.size),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def lexical_recall_prev(utt: pd.DataFrame) -> np.ndarray:
    """Per therapist content turn: share of the preceding patient turn's content words (stopwords
    out, ≥3 chars) that recur in the therapist turn — the lexical twin of ``echo``. A reflection
    restates the patient's words, so a reflection-heavy policy should score high; NaN elsewhere."""
    _check_aligned(utt)
    out = np.full(len(utt), np.nan)
    th = content_mask(utt, "therapist")
    pos = {(a, it, fi, ui): i for i, (a, it, fi, ui) in
           enumerate(zip(utt["arm"], utt["iteration"], utt["file_index"], utt["utt_idx"]))}
    texts = utt["text"].astype(str).to_numpy()
    for i in np.flatnonzero(th):
        j = pos.get((utt.at[i, "arm"], utt.at[i, "iteration"], utt.at[i, "file_index"], utt.at[i, "utt_idx"] - 1))
        if j is None or utt.at[j, "role"] != "patient":
            continue
        prev = {w for w in _TOK.findall(texts[j].lower()) if w not in _STOP and len(w) > 2}
        if not prev:
            continue
        cur = {w for w in _TOK.findall(texts[i].lower())}
        out[i] = len(prev & cur) / len(prev)
    return out


def profile_kcontrast(prof_conv: pd.DataFrame, arms, *, features: Sequence[str] = ("n_chars", "q_count", "effusive", "echo"),
                      iteration: Optional[int] = None) -> pd.DataFrame:
    """Persona-paired K=0 − K=5 per (method, turn bin, feature) at one iteration (default: the last
    iteration both K arms of the method share). Holm across bins within (method, feature)."""
    from .behavior import _attach_by_arm
    from .stats import paired_arrays, holm
    df = prof_conv if "persona_id" in prof_conv.columns else _attach_by_arm(prof_conv, arms)
    rows = []
    for method in ("PTO", "GRPO"):
        a0, a5 = df[df["arm"] == f"{method}_LA0"], df[df["arm"] == f"{method}_LA5"]
        if a0.empty or a5.empty:
            continue
        common = sorted(set(a0["iteration"]) & set(a5["iteration"]))
        if not common:
            continue
        it = iteration if iteration in common else common[-1]
        b0, b5 = a0[a0["iteration"] == it], a5[a5["iteration"] == it]
        for feat in features:
            for _, _, lab in PROFILE_BINS:
                x = b0[b0["bin"] == lab][["persona_id", feat]].merge(
                    b5[b5["bin"] == lab][["persona_id", feat]], on="persona_id", suffixes=("_0", "_5"))
                r = paired_arrays(x[f"{feat}_0"], x[f"{feat}_5"])
                rows.append({"method": method, "iteration": it, "feature": feat, "bin": lab,
                             "mean_K0": float(x[f"{feat}_0"].mean()) if len(x) else np.nan,
                             "mean_K5": float(x[f"{feat}_5"].mean()) if len(x) else np.nan, **r})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for (m, f), g in out.groupby(["method", "feature"]):
        ok = g["p"].notna()
        if ok.any():
            out.loc[g.index[ok], "p_holm"] = holm(g.loc[ok, "p"].to_numpy())
    return out


def diversity_by_state(utt: pd.DataFrame, E: np.ndarray, *, max_turn: int = 8,
                       dup_threshold: float = 0.95, n_boot: int = 300,
                       sample_utts: int = 400, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Per (arm, iteration) diversity scalars over the therapist's content turns.

    ``template_sim``: mean pairwise cosine across conversations at the same ``role_turn``
    (1..``max_turn``), averaged over turns — high = every persona gets the same turn.
    ``persona_var_share``: between-conversation share of total embedding variance (+ bootstrap CI
    over conversations) — low = the therapist says the same things whoever the patient is.
    ``dup_rate``: share of turns whose nearest neighbour in ANOTHER conversation of the state has
    cosine ≥ ``dup_threshold``. ``distinct_{1,2,3}``: unique / total n-grams over a fixed random
    sample of ``sample_utts`` turns (pooled distinct-n is length-confounded; the sample is not).
    """
    rng = np.random.default_rng(seed)
    mask = content_mask(utt)
    rows = []
    for key, g in utt[mask].groupby(_STATE, sort=False):
        gi = g.index.to_numpy()
        X = E[gi]
        conv = g["file_index"].to_numpy()
        # template similarity at matched turn index
        sims = []
        for r in range(1, max_turn + 1):
            sel = (g["role_turn"].to_numpy() == r)
            if sel.sum() >= 2:
                sims.append(_mean_pairwise_cos(X[sel]))
        # between-conversation variance share
        xbar = X.mean(axis=0)
        total = float(((X - xbar) ** 2).sum())
        convs = np.unique(conv)
        cents = {c: X[conv == c].mean(axis=0) for c in convs}
        ns = {c: int((conv == c).sum()) for c in convs}
        between = float(sum(ns[c] * ((cents[c] - xbar) ** 2).sum() for c in convs))
        share = between / total if total > 0 else np.nan
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(convs, size=convs.size, replace=True)
            idx = np.concatenate([np.flatnonzero(conv == c) for c in pick])
            Xb = X[idx]; xb = Xb.mean(axis=0)
            tot_b = float(((Xb - xb) ** 2).sum())
            bet_b = float(sum(ns[c] * ((cents[c] - xb) ** 2).sum() for c in pick))
            boots.append(bet_b / tot_b if tot_b > 0 else np.nan)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        # near-duplicate rate across conversations
        S = X @ X.T
        same = conv[:, None] == conv[None, :]
        S[same] = -1.0
        dup = float((S.max(axis=1) >= dup_threshold).mean()) if len(X) > 1 else np.nan
        # distinct-n on a fixed-size sample
        texts = g["text"].tolist()
        pick = rng.choice(len(texts), size=min(sample_utts, len(texts)), replace=False)
        toks = [_TOK.findall(texts[i].lower()) for i in pick]
        dist = {}
        for n in (1, 2, 3):
            grams = [tuple(t[j:j + n]) for t in toks for j in range(len(t) - n + 1)]
            dist[f"distinct_{n}"] = len(set(grams)) / len(grams) if grams else np.nan
        row = dict(zip(_STATE, key))
        row.update({"template_sim": float(np.mean(sims)) if sims else np.nan,
                    "persona_var_share": share, "persona_var_share_lo": float(lo),
                    "persona_var_share_hi": float(hi), "dup_rate": dup, **dist,
                    "n_turns": int(len(X)), "n_convs": int(convs.size),
                    "mean_words": float(g["n_words"].mean())})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "iteration"]).reset_index(drop=True)


def template_similarity_by_turn(utt: pd.DataFrame, E: np.ndarray, *, max_turn: int = 10) -> pd.DataFrame:
    """Long table: per (arm, iteration, role_turn) mean pairwise cosine across conversations."""
    mask = content_mask(utt)
    rows = []
    for key, g in utt[mask].groupby(_STATE, sort=False):
        for r in range(1, max_turn + 1):
            sel = g.index.to_numpy()[g["role_turn"].to_numpy() == r]
            if sel.size >= 2:
                row = dict(zip(_STATE, key))
                row.update({"role_turn": r, "template_sim": _mean_pairwise_cos(E[sel]),
                            "n": int(sel.size)})
                rows.append(row)
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  A5 · Within-session profile                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _bin_of(rt: int) -> str:
    for lo, hi, lab in PROFILE_BINS:
        if lo <= rt <= hi:
            return lab
    return "0"


def _therapist_turns_with_features(utt: pd.DataFrame, feats: Dict[str, np.ndarray]) -> pd.DataFrame:
    th = utt[content_mask(utt)].copy()
    for col in ("echo", "lex_recall_prev"):
        arr = feats.get(col)
        th[col] = arr[th.index.to_numpy()] if arr is not None else np.nan
    th["effusive"] = th["effusive"].astype(float)
    th["affirm"] = th["affirm"].astype(float)
    th["bin"] = th["role_turn"].map(_bin_of)
    return th


def session_profile(utt: pd.DataFrame, feats: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per (arm, iteration, turn bin): mean therapist :data:`PROFILE_FEATURES` and the share of
    conversations still running at that bin. ``feats`` = :func:`utterance_features`."""
    th = _therapist_turns_with_features(utt, feats)
    n_conv = th.groupby(_STATE, sort=False)["file_index"].nunique().rename("n_convs_state")
    agg = (th.groupby(_STATE + ["bin"], sort=False)
           .agg(**{f: (f, "mean") for f in PROFILE_FEATURES},
                n_turns=("n_chars", "size"), n_convs=("file_index", "nunique")).reset_index())
    agg = agg.merge(n_conv.reset_index(), on=_STATE)
    agg["share_convs_reaching"] = agg["n_convs"] / agg["n_convs_state"]
    order = {lab: i for i, (_, _, lab) in enumerate(PROFILE_BINS)}
    agg["bin_order"] = agg["bin"].map(order)
    return agg.sort_values(["arm", "iteration", "bin_order"]).drop(columns="bin_order").reset_index(drop=True)


def profile_per_conv(utt: pd.DataFrame, feats: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per (conversation, turn bin) means of the same features — for persona-paired bin contrasts."""
    th = _therapist_turns_with_features(utt, feats)
    return (th.groupby(_CONV + ["bin"], sort=False)
            .agg(**{f: (f, "mean") for f in PROFILE_FEATURES}, n_turns=("n_chars", "size"))
            .reset_index())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Glue — scores_long adapters, echo validation, number ledger                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def to_scores_long(per_conv: pd.DataFrame, arms, metrics: Sequence[str]) -> pd.DataFrame:
    """Reshape a per-conversation frame into the ``load_scores_long`` schema so every ``stats``
    entry point (``paired_k_comparison``, ``k_means_by_iter``, …) applies unchanged."""
    from .behavior import _attach_by_arm
    oracle_by_arm = {a.label: a.oracle for a in arms}
    df = per_conv.copy()
    if "persona_id" not in df.columns:
        df = _attach_by_arm(df, arms)
    df["oracle"] = df["arm"].map(oracle_by_arm)
    id_cols = ["method", "arm", "K", "oracle", "model", "iteration", "is_base", "file_index",
               "persona_id"] + [c for c in PERSONA_COLS if c in df.columns]
    value_cols = [m for m in metrics if m in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="questionnaire", value_name="score")
    return long.dropna(subset=["score"]).reset_index(drop=True)


def echo_validation(per_conv: pd.DataFrame, miti: pd.DataFrame,
                    proxies: Sequence[str] = ("echo", "lex_recall_prev")) -> pd.DataFrame:
    """Do the judge-free responsiveness proxies track the oracle's reflection counts?

    Per (arm, iteration, proxy): Spearman ρ across the 96 conversations between the proxy and
    MITI reflections per therapist turn (``(B4_SR + B5_CR) / n_th_turns``), plus the same against
    ``B3_Q``-per-turn as a discriminant check (a reflection proxy should NOT track questions), and
    against complex reflections alone. ``miti`` is :func:`behavior.load_miti_behavior` under one
    grader; the caller names the grader. A pooled within-state ρ (Fisher-z mean over states) is
    the number to quote.
    """
    from scipy.stats import spearmanr
    m = miti.copy()
    keys = ["arm", "iteration", "file_index"]
    m["refl"] = (m["B4_SR"].fillna(0) + m["B5_CR"].fillna(0))
    m["cr"] = m["B5_CR"].fillna(0)
    m["q"] = m["B3_Q"]
    j = per_conv.merge(m[keys + ["refl", "cr", "q"]], on=keys, how="inner")
    den = j["n_th_turns"].replace(0, np.nan)
    for c in ("refl", "cr", "q"):
        j[c] = j[c] / den
    rows = []
    for proxy in proxies:
        if proxy not in j.columns:
            continue
        for key, g in j.groupby(_STATE, sort=False):
            g = g.dropna(subset=[proxy, "refl"])
            if len(g) < 10:
                continue
            row = dict(zip(_STATE, key))
            row["proxy"] = proxy
            row["n"] = int(len(g))
            for tgt, lab in (("refl", "reflections"), ("cr", "complex_reflections"), ("q", "questions")):
                gg = g.dropna(subset=[tgt])
                r = spearmanr(gg[proxy], gg[tgt]) if len(gg) >= 10 and gg[tgt].nunique() > 1 else None
                row[f"rho_vs_{lab}"] = float(r.correlation) if r is not None else np.nan
                if lab == "reflections":
                    row["p_vs_reflections"] = float(r.pvalue) if r is not None else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["proxy", "arm", "iteration"]).reset_index(drop=True)


def pooled_rho(val: pd.DataFrame, cols: Sequence[str] = ("rho_vs_reflections", "rho_vs_complex_reflections",
                                                        "rho_vs_questions")) -> pd.DataFrame:
    """Fisher-z pooled within-state Spearman ρ per (proxy, method) with an n-weighted mean."""
    rows = []
    for (proxy, method), g in val.groupby(["proxy", "method"]):
        row = {"proxy": proxy, "method": method, "n_states": int(len(g))}
        for c in cols:
            z = np.arctanh(np.clip(g[c].to_numpy(float), -0.999, 0.999))
            w = g["n"].to_numpy(float)
            ok = np.isfinite(z)
            row[c] = float(np.tanh((z[ok] * w[ok]).sum() / w[ok].sum())) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def state_table(per_conv: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Per (arm, iteration) mean ± SE of per-conversation metrics (the LEVEL table)."""
    g = per_conv.groupby(_STATE, sort=False)
    out = g[list(metrics)].mean()
    se = g[list(metrics)].sem().add_suffix("_se")
    n = g.size().rename("n")
    return pd.concat([out, se, n], axis=1).reset_index().sort_values(["arm", "iteration"]).reset_index(drop=True)


def text_numbers(state_rep: pd.DataFrame, drift: pd.DataFrame, cosines: pd.DataFrame,
                 div: pd.DataFrame, levels: pd.DataFrame, rep: dict) -> Dict[str, dict]:
    """The ledger: endpoint scalars per arm with their source table named."""
    out = {}
    out["repertoire.k"] = {"value": rep["k"], "source": "repertoire_fit", "note": "k-means clusters on the base pool"}
    out["repertoire.novel_threshold"] = {"value": round(rep["threshold"], 4), "source": "repertoire_fit",
                                         "note": f"{rep['novel_pct']}th pct of held-out base nearest-centroid cosine"}
    for a, g in state_rep.groupby("arm"):
        g = g.sort_values("iteration")
        f = g.iloc[-1]
        out[f"repertoire.{a}.final_iter"] = {"value": int(f["iteration"]), "source": "repertoire_by_state", "note": ""}
        for c in ("novel_share", "entropy", "eff_clusters"):
            out[f"repertoire.{a}.{c}.base"] = {"value": round(float(g.iloc[0][c]), 4), "source": "repertoire_by_state", "note": "iteration 0"}
            out[f"repertoire.{a}.{c}.final"] = {"value": round(float(f[c]), 4), "source": "repertoire_by_state", "note": f"iteration {int(f['iteration'])}"}
    for a, g in drift.groupby("arm"):
        f = g.sort_values("iteration").iloc[-1]
        out[f"drift.{a}.drift_norm.final"] = {"value": round(float(f["drift_norm"]), 4), "source": "drift_by_state", "note": "vs pooled base centroid"}
    if not cosines.empty:
        f = cosines.sort_values("iteration").iloc[-1]
        for c in cosines.columns:
            if c.startswith("cos_"):
                out[f"drift.{c}.final"] = {"value": round(float(f[c]), 4), "source": "drift_cosines", "note": f"iteration {int(f['iteration'])}"}
    for a, g in div.groupby("arm"):
        g = g.sort_values("iteration")
        for c in ("template_sim", "persona_var_share", "dup_rate", "distinct_2"):
            out[f"diversity.{a}.{c}.base"] = {"value": round(float(g.iloc[0][c]), 4), "source": "diversity_by_state", "note": "iteration 0"}
            out[f"diversity.{a}.{c}.final"] = {"value": round(float(g.iloc[-1][c]), 4), "source": "diversity_by_state", "note": f"iteration {int(g.iloc[-1]['iteration'])}"}
    for a, g in levels.groupby("arm"):
        g = g.sort_values("iteration")
        for c in TEXT_K_METRICS:
            if c in g.columns:
                out[f"levels.{a}.{c}.base"] = {"value": round(float(g.iloc[0][c]), 4), "source": "text_levels", "note": "iteration 0"}
                out[f"levels.{a}.{c}.final"] = {"value": round(float(g.iloc[-1][c]), 4), "source": "text_levels", "note": f"iteration {int(g.iloc[-1]['iteration'])}"}
    return out
