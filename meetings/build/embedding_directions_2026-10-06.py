"""
embedding_directions_2026-10-06.py -- Doron's win/lose-direction analysis (his October deck "Win/Lose
Direction: Does the Embedding Model Matter?") run in five embedding spaces and read through words and
sentences, with the 2026-10-05 checks built in. Lior, 2026-10-06: "lets run the embedding eda like doron
did (also words and also sentences with multiple embedders) and think how it can fit the paper".

Data: the GRPO rounds of the Doron share (``grpo_groups/``; ``train`` split, rounds whose scored
candidates have at least two different rewards), exactly the rounds of the 2026-09-29 and 2026-10-05
analyses. A round's 8 candidates answer one conversation-so-far, so every direction is WITHIN a round.

Cells: arm (GRPO_K0, GRPO_K5) x training iteration 1-10 (iteration n samples from the model of n-1).

Two estimators of a cell's direction (both unit norm):
  * ``winlose`` -- Doron's: the mean over rounds of emb(best) - emb(worst), tied extremes averaged;
  * ``advw``    -- the paper's Appendix C.5 proxy: the sum over candidates of w * emb, w the reward's
                   deviation from its round mean rescaled to sum |w| = 2 within the round
                   (= eda_analysis.pref's GRPO weight; the advantage's SD cancels in the rescaling).
Two candidate sets: ``all``, and ``clean`` = no leaked chat marker and no degenerate text, with best,
worst and the weights re-computed among the kept candidates.

Five embedders: all-MiniLM-L6-v2 (the paper's App C.5 estimator), gte-base-en-v1.5 (the 2026-09-29 and
10-05 runs; candidate embeddings reused from the 09-29 cache), Qwen3-Embedding-0.6B and
mxbai-embed-large-v1 (Doron's two), and Llama-3.2-1B, the therapist's own base: the layer-8 output
(hidden_states[8] of 16, before the final norm) mean-pooled over the text's tokens, BOS EXCLUDED (a
mid-layer BOS activation is orders of magnitude larger than any other token's and would dominate the
mean; Doron's deck pooled "all real tokens").

Per embedder (and estimator x candidate set):
  1. cosines -- K0 vs K5 per iteration and consecutive iterations per arm, raw and noise-corrected
     (50 patient split-halves; corrected = cross-half cosine / sqrt(rel_1 * rel_2)), plus Doron's
     "is a round representative" = mean cos(round's own vector, its cell's direction);
  2. sentences -- Doron's pool (sentences of 4-10 words seen >= 3 times verbatim, the 4,488 most
     frequent) projected on every direction: top / bottom sentences per cell and for K0 - K5;
  3. sentence CATEGORIES, FIXED BEFORE THIS RUN (2026-10-06, see CATEGORIES): per direction,
     (mean projection of the category's pool sentences - mean of the rest) / SD of the pool, with a
     95% interval from 500 patient bootstraps (the same draws in both arms, so K0 - K5 is paired);
  4. words -- Doron's general vocabulary (the 20,000 most frequent English words of >= 3 letters,
     ``wordfreq``) and the corpus vocabulary (words in >= 100 candidates), each word embedded alone;
  5. agreement BETWEEN embedders -- Spearman of the pool projections per cell (spaces differ, so
     cosines cannot be compared across embedders; rankings of the same sentences can).
Embedding-free, once: 6. lexical log-odds (Monroe, Colaresi and Quinn 2008, informative Dirichlet
prior) of word counts in the best vs the worst candidates of each cell (clean set; ties weighted
1/n): which words win within a round, with no model in the loop.

VRAM: each model runs under torch.cuda.set_per_process_memory_fraction(0.5) with length-sorted,
bounded batches, so an overrun raises OutOfMemoryError instead of rebooting the PC (12 GB local card).

Usage (repo root, repo .venv):
    .venv/Scripts/python.exe meetings/build/embedding_directions_2026-10-06.py --smoke
    .venv/Scripts/python.exe meetings/build/embedding_directions_2026-10-06.py [--embedders minilm,gte,mxbai,qwen3,llama8]
"""

import argparse
import collections
import gc
import gzip
import importlib.util
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch  # noqa: F401  # before transformers (local sm_120 import order)
from scipy import sparse
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rd = _load("round_directions", "analyze_grpo_round_directions_2026-09-29.py")
chk = _load("doron_check", "check_doron_directions_2026-10-05.py")

DEFAULT_OUT = os.path.join(REPO, "meetings", "2026-10-06_embedding_directions")
GTE_SEED_CACHE = os.path.join(rd.DEFAULT_OUT, ".emb_cache")     # 09-29 candidate embeddings (gte)
ARMS, ITERS, CELLS, LAB, COL = rd.ARMS, rd.ITERS, rd.CELLS, rd.LAB, rd.COL
N_SPLITS, N_BOOT, SEED = 50, 500, 0
TOP_S, TOP_W = 3, 8
LLAMA = "meta-llama/Llama-3.2-1B"

EMBEDDERS = {   # key: (label, kind, model id, dtype, max tokens, batch)
    "minilm": ("all-MiniLM-L6-v2", "st", "sentence-transformers/all-MiniLM-L6-v2", torch.float16, 256, 128),
    "gte": ("gte-base-en-v1.5", "gte", rd.MODEL, None, 512, None),
    "mxbai": ("mxbai-embed-large-v1", "st", "mixedbread-ai/mxbai-embed-large-v1", torch.float16, 512, 64),
    "qwen3": ("Qwen3-Embedding-0.6B", "st", "Qwen/Qwen3-Embedding-0.6B", torch.bfloat16, 512, 64),
    "llama8": ("Llama-3.2-1B layer 8", "llama", LLAMA, torch.bfloat16, 512, None),
}
MARK = {"minilm": "o", "gte": "s", "mxbai": "^", "qwen3": "D", "llama8": "v"}

# Sentence categories, fixed on 2026-10-06 BEFORE this run (informed by the 10-05 reads and by the
# utterance coder's code families: praise / thanks / agreement ~ PRA, question ~ OQ+CQ, reflection
# opener ~ SR+CR, advice ~ PERS+GI, list preamble = the templated-list artefact). Matched on the
# pool sentence, case-insensitive; a sentence may fall in several.
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
_WORDTOK = re.compile(r"[a-z]+(?:'[a-z]+)?")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


# ── encoders ──────────────────────────────────────────────────────────────────────────────────
def _cap_vram():
    torch.cuda.set_per_process_memory_fraction(0.5)


def st_encode(model_id, dtype, max_len, batch):
    def enc(texts):
        from sentence_transformers import SentenceTransformer
        _cap_vram()
        m = SentenceTransformer(model_id, device="cuda", model_kwargs={"dtype": dtype})
        m.max_seq_length = max_len
        E = m.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=len(texts) > 5000)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return E.astype(np.float16)
    return enc


def gte_encode(texts):
    _cap_vram()
    return rd._encode(texts)


def llama_encode(texts, layer=8, max_batch=128, token_budget=12288):
    from transformers import AutoModel, AutoTokenizer
    _cap_vram()
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    m = AutoModel.from_pretrained(LLAMA, dtype=torch.bfloat16)
    m.layers = m.layers[:layer]                      # hidden_states[layer] = the output of decoder layer `layer`
    m.norm = torch.nn.Identity()                     # ... before the final RMSNorm, which only the last layer gets
    m.config.num_hidden_layers = layer
    m = m.cuda().eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), m.config.hidden_size), dtype=np.float16)
    i, t0, nb = 0, time.time(), 0
    while i < len(order):
        longest = len(texts[order[min(i + max_batch, len(order)) - 1]]) // 3 + 2   # chars/3 over-estimates tokens
        n = max(1, min(max_batch, token_budget // max(1, min(512, longest))))
        idx = order[i:i + n]
        x = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = m(input_ids=x.input_ids, attention_mask=x.attention_mask, use_cache=False).last_hidden_state.float()
        mask = x.attention_mask.clone().float()
        mask[:, 0] = 0.0                              # right padding: position 0 is always BOS
        v = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1.0).unsqueeze(-1)
        out[idx] = torch.nn.functional.normalize(v, dim=1).cpu().numpy()
        i += n
        nb += 1
        if nb % 300 == 0:
            print(f"    llama8 {i:,}/{len(order):,}  {time.time() - t0:.0f}s  "
                  f"vram {torch.cuda.memory_reserved() / 2**30:.1f}G", flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


def encoder_for(key):
    label, kind, mid, dtype, max_len, batch = EMBEDDERS[key]
    if kind == "st":
        return st_encode(mid, dtype, max_len, batch)
    if kind == "gte":
        return gte_encode
    return llama_encode


def embed_cached(key, texts, out):
    """Unit-norm float32 embeddings of ``texts`` under embedder ``key``, cached by text."""
    cdir = os.path.join(out, ".emb_cache", key)
    tpath, epath = os.path.join(cdir, "texts.json.gz"), os.path.join(cdir, "emb.npy")
    known, E = [], None
    if os.path.exists(tpath):
        known = json.load(gzip.open(tpath, "rt", encoding="utf-8"))
        E = np.load(epath)
    elif key == "gte" and os.path.exists(os.path.join(GTE_SEED_CACHE, "gte_texts.json.gz")):
        known = json.load(gzip.open(os.path.join(GTE_SEED_CACHE, "gte_texts.json.gz"), "rt", encoding="utf-8"))
        E = np.load(os.path.join(GTE_SEED_CACHE, "gte_emb.npy"))
        print(f"  gte: seeded from the 2026-09-29 cache ({len(known):,} texts)", flush=True)
    pos = {t: k for k, t in enumerate(known)}
    missing = sorted({t for t in texts if t not in pos})
    if missing:
        print(f"  {key}: embedding {len(missing):,} new texts ...", flush=True)
        t0 = time.time()
        new = encoder_for(key)(missing)
        print(f"  {key}: {len(missing):,} texts in {time.time() - t0:.0f}s", flush=True)
        E = new if E is None else np.vstack([E, new])
        known += missing
        pos.update({t: len(known) - len(missing) + k for k, t in enumerate(missing)})
        os.makedirs(cdir, exist_ok=True)
        np.save(epath, E)
        with gzip.open(tpath, "wt", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
    X = E[[pos[t] for t in texts]].astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)


# ── round vectors ─────────────────────────────────────────────────────────────────────────────
def round_vectors(D, E, keep, estimator):
    """Per round, the vector its candidates contribute to the cell direction, and whether it counts."""
    R, cr = len(D.rounds), D.cand_round
    starts = D.rounds.start.to_numpy()
    rmax = np.maximum.reduceat(np.where(keep, D.r, -np.inf), starts)
    rmin = np.minimum.reduceat(np.where(keep, D.r, np.inf), starts)
    valid = rmax > rmin
    w = np.zeros(len(D.r))
    if estimator == "winlose":
        best = keep & valid[cr] & (D.r == rmax[cr])
        worst = keep & valid[cr] & (D.r == rmin[cr])
        nb, nw = np.bincount(cr[best], minlength=R), np.bincount(cr[worst], minlength=R)
        w[best], w[worst] = 1.0 / nb[cr[best]], -1.0 / nw[cr[worst]]
    else:   # advw
        k = keep & valid[cr]
        cnt = np.bincount(cr[k], minlength=R)
        mean = np.bincount(cr[k], weights=D.r[k], minlength=R) / np.maximum(cnt, 1)
        dev = np.where(k, D.r - mean[cr], 0.0)
        tot = np.bincount(cr, weights=np.abs(dev), minlength=R)
        w = np.where(k, 2.0 * dev / np.maximum(tot[cr], 1e-12), 0.0)
    nz = np.flatnonzero(w)
    M = sparse.csr_matrix((w[nz], (cr[nz], nz)), shape=(R, len(D.r)))
    return np.asarray(M @ E, dtype=np.float64), valid, w


class Cells:
    """Per cell: per-patient sums of round vectors (P x d), the full direction, 50 split-half pairs."""

    def __init__(self, D, V, valid, splits):
        self.P = int(D.rounds.patient.max()) + 1
        arm, it, pat = D.rounds.arm.to_numpy(), D.rounds.iteration.to_numpy(), D.rounds.patient.to_numpy()
        self.S, self.full, self.halves, self.n_rounds, self.repr = {}, {}, {}, {}, {}
        for cell in CELLS:
            idx = np.flatnonzero(valid & (arm == cell[0]) & (it == cell[1]))
            S = np.zeros((self.P, V.shape[1]))
            np.add.at(S, pat[idx], V[idx])
            self.S[cell], self.n_rounds[cell] = S, len(idx)
            self.full[cell] = unit(S.sum(0))
            self.halves[cell] = np.stack([unit(np.stack([S[a].sum(0), S[b].sum(0)])) for a, b in splits])
            rc = unit(V[idx]) @ self.full[cell]
            self.repr[cell] = (float(rc.mean()), float(rc.std()))

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


def cosine_rows(C, emb_key, variant, estimator):
    rows = []
    for arm in ARMS:
        for it in ITERS:
            c = (arm, it)
            row = dict(embedder=emb_key, variant=variant, estimator=estimator, arm=arm, iteration=it,
                       n_rounds=C.n_rounds[c], reliability=C.rel(c),
                       round_repr_mean=C.repr[c][0], round_repr_sd=C.repr[c][1])
            if it > 1:
                row["stability_raw"] = C.cos(c, (arm, it - 1))
                row["stability_corrected"] = C.cos_corrected(c, (arm, it - 1))
            if arm == "GRPO_K0":
                row["k0_vs_k5_raw"] = C.cos(c, ("GRPO_K5", it))
                row["k0_vs_k5_corrected"] = C.cos_corrected(c, ("GRPO_K5", it))
            rows.append(row)
    return rows


def category_rows(C, Ep, cat_masks, boot, emb_key, variant, estimator):
    """z of each category per direction (and K0 - K5), with a 95% patient-bootstrap interval."""
    def zs(W):                                     # W: (d, m) unit directions -> (n_cat, m)
        p = Ep @ W
        mu, sd = p.mean(0), p.std(0)
        return np.stack([(p[m].mean(0) - p[~m].mean(0)) / sd for m in cat_masks.values()])

    rows = []
    for it in ITERS:
        per_arm = {}
        for arm in ARMS:
            S = C.S[(arm, it)]
            Wb = unit(boot @ S).T                  # (d, B)
            per_arm[arm] = (zs(C.full[(arm, it)][:, None])[:, 0], zs(Wb))
        for side in ARMS + ["K0-K5"]:
            if side == "K0-K5":
                pt = per_arm["GRPO_K0"][0] - per_arm["GRPO_K5"][0]
                bs = per_arm["GRPO_K0"][1] - per_arm["GRPO_K5"][1]
            else:
                pt, bs = per_arm[side]
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=1)
            for k, cat in enumerate(cat_masks):
                rows.append(dict(embedder=emb_key, variant=variant, estimator=estimator, side=side, iteration=it,
                                 category=cat, z=float(pt[k]), lo=float(lo[k]), hi=float(hi[k])))
    return rows


# ── lexical log-odds (no embedder) ───────────────────────────────────────────────────────────
def lexical_logodds(D, keep, prior_total=1000.0, min_count=20):
    """Monroe et al. (2008) log-odds with an informative Dirichlet prior, best vs worst per cell."""
    R, cr = len(D.rounds), D.cand_round
    starts = D.rounds.start.to_numpy()
    rmax = np.maximum.reduceat(np.where(keep, D.r, -np.inf), starts)
    rmin = np.minimum.reduceat(np.where(keep, D.r, np.inf), starts)
    valid = rmax > rmin
    best = keep & valid[cr] & (D.r == rmax[cr])
    worst = keep & valid[cr] & (D.r == rmin[cr])
    nb, nw = np.bincount(cr[best], minlength=R), np.bincount(cr[worst], minlength=R)
    toks = [_WORDTOK.findall(t.lower()) for t in D.text]
    bg = collections.Counter()
    for k in np.flatnonzero(keep):
        bg.update(toks[k])
    vocab = [w for w, c in bg.items() if c >= min_count]
    vid = {w: i for i, w in enumerate(vocab)}
    bgv = np.array([bg[w] for w in vocab], float)
    alpha = prior_total * bgv / bgv.sum()
    a0 = alpha.sum()
    arm, it = D.rounds.arm.to_numpy(), D.rounds.iteration.to_numpy()
    rows = []
    for cell in CELLS + [(a, "4-10") for a in ARMS] + [(a, "1-3") for a in ARMS]:
        if isinstance(cell[1], str):
            lo_, hi_ = map(int, cell[1].split("-"))
            rsel = (arm == cell[0]) & (it >= lo_) & (it <= hi_)
        else:
            rsel = (arm == cell[0]) & (it == cell[1])
        yb, yw = np.zeros(len(vocab)), np.zeros(len(vocab))
        for side, msk, nn, y in (("b", best, nb, yb), ("w", worst, nw, yw)):
            for k in np.flatnonzero(msk & rsel[cr]):
                wt = 1.0 / nn[cr[k]]
                for t in toks[k]:
                    j = vid.get(t)
                    if j is not None:
                        y[j] += wt
        n1, n2 = yb.sum(), yw.sum()
        d = (np.log((yb + alpha) / (n1 + a0 - yb - alpha)) - np.log((yw + alpha) / (n2 + a0 - yw - alpha)))
        z = d / np.sqrt(1.0 / (yb + alpha) + 1.0 / (yw + alpha))
        for j in np.argsort(-z)[:15].tolist() + np.argsort(z)[:15].tolist():
            rows.append(dict(arm=cell[0], iteration=str(cell[1]), word=vocab[j], z=float(z[j]),
                             best_count=float(yb[j]), worst_count=float(yw[j])))
    return pd.DataFrame(rows)


# ── output helpers ────────────────────────────────────────────────────────────────────────────
def md(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt(items):
    return "; ".join(f"“{md(s)}”" for s in items)


def sentence_tables(C, Ep, S, emb_key, label):
    lines = [f"# {label}: top win / lose pool sentences (clean candidates, winlose)\n",
             "Pool restricted to sentences without a chat marker. win = highest projection on the cell's",
             "direction, lose = lowest; K0 − K5 = the difference of the two arms' directions.\n",
             "| iter | K=0 win | K=0 lose | K=5 win | K=5 lose | more K=0 | more K=5 |", "|---|---|---|---|---|---|---|"]
    rows = []
    for it in ITERS:
        cells = []
        for arm in ARMS:
            p = Ep @ C.full[(arm, it)]
            o = np.argsort(-p)
            cells += [fmt(S[o[:TOP_S]]), fmt(S[o[::-1][:TOP_S]])]
            rows += [dict(embedder=emb_key, iteration=it, direction=LAB[arm], side=s, rank=r + 1, sentence=S[j])
                     for s, oo in (("win", o), ("lose", o[::-1])) for r, j in enumerate(oo[:10])]
        p = Ep @ (C.full[("GRPO_K0", it)] - C.full[("GRPO_K5", it)])
        o = np.argsort(-p)
        cells += [fmt(S[o[:TOP_S]]), fmt(S[o[::-1][:TOP_S]])]
        rows += [dict(embedder=emb_key, iteration=it, direction="K0-K5", side=s, rank=r + 1, sentence=S[j])
                 for s, oo in (("more_K0", o), ("more_K5", o[::-1])) for r, j in enumerate(oo[:10])]
        lines.append(f"| {it} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", rows


def word_tables(C, Ew, words, emb_key, label, vocab_name):
    lines = [f"\n## {label}: top words, {vocab_name} vocabulary (clean candidates, winlose)\n",
             "| iter | K=0 win | K=0 lose | K=5 win | K=5 lose | more K=0 | more K=5 |", "|---|---|---|---|---|---|---|"]
    for it in ITERS:
        cells = []
        for d in (C.full[("GRPO_K0", it)], C.full[("GRPO_K5", it)], C.full[("GRPO_K0", it)] - C.full[("GRPO_K5", it)]):
            o = np.argsort(-(Ew @ d))
            cells += [", ".join(words[o[:TOP_W]]), ", ".join(words[o[::-1][:TOP_W]])]
        lines.append(f"| {it} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ── figures ───────────────────────────────────────────────────────────────────────────────────
def figures(cos, cat, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(out, "figures")
    os.makedirs(fdir, exist_ok=True)
    plt.rcParams.update({"font.size": 8})
    for est in ("winlose", "advw"):
        c = cos[(cos.variant == "clean") & (cos.estimator == est)]
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
        for k, (e, g) in enumerate(c.groupby("embedder", sort=False)):
            lab = EMBEDDERS[e][0]
            k0 = g[g.arm == "GRPO_K0"].sort_values("iteration")
            ax[0].plot(k0.iteration, k0.k0_vs_k5_corrected, marker=MARK[e], ms=3, lw=1, label=lab)
            for j, arm in enumerate(ARMS):
                ga = g[(g.arm == arm) & g.stability_corrected.notna()].sort_values("iteration")
                ax[1 + j].plot(ga.iteration, ga.stability_corrected, marker=MARK[e], ms=3, lw=1, label=lab)
        ax[0].set_title("(a) K=0 vs K=5, same iteration", loc="left")
        ax[1].set_title("(b) K=0, iteration n vs n−1", loc="left")
        ax[2].set_title("(c) K=5, iteration n vs n−1", loc="left")
        for a in ax:
            a.axhline(0, color="#999", lw=0.6)
            a.set_xticks(ITERS)
            a.set_xlabel("training iteration")
            a.set_ylim(-1.0, 1.25)
        ax[0].set_ylabel("noise-corrected cosine")
        ax[0].legend(frameon=False, fontsize=6, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, f"cosines_{est}.png"), dpi=200)
        plt.close(fig)

    cats = list(CATEGORIES)
    c = cat[(cat.variant == "clean") & (cat.estimator == "winlose")]
    fig, axes = plt.subplots(3, len(cats), figsize=(2.0 * len(cats), 6.2), sharex=True, sharey="row")
    for col, cg in enumerate(cats):
        for row, side in enumerate(ARMS + ["K0-K5"]):
            ax = axes[row, col]
            for e, g in c[(c.category == cg) & (c.side == side)].groupby("embedder", sort=False):
                g = g.sort_values("iteration")
                color = COL.get(side, "#444")
                ax.plot(g.iteration, g.z, marker=MARK[e], ms=2.5, lw=0.9, color=color, alpha=0.85,
                        label=EMBEDDERS[e][0])
            ax.axhline(0, color="#999", lw=0.6)
            if row == 0:
                ax.set_title(cg)
            if col == 0:
                ax.set_ylabel({"GRPO_K0": "K=0 direction", "GRPO_K5": "K=5 direction", "K0-K5": "K=0 − K=5"}[side]
                              + "\n(z, category vs rest)")
            if row == 2:
                ax.set_xticks([1, 4, 7, 10])
                ax.set_xlabel("training iteration")
    axes[0, 0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "categories_winlose.png"), dpi=200)
    plt.close(fig)


# ── smoke ─────────────────────────────────────────────────────────────────────────────────────
def smoke():
    texts = ["I'm so proud of you for taking this step.", "What would you like to change first?",
             "It sounds like you feel stuck.", "Here are the three steps:", "proud", "x" * 900]
    for key in EMBEDDERS:
        t0 = time.time()
        E = encoder_for(key)(texts)
        E = E.astype(np.float32)
        print(f"{key}: shape {E.shape}, norms {np.round(np.linalg.norm(E, axis=1), 3)}, "
              f"cos(0,1) {float(unit(E[0]) @ unit(E[1])):.3f}, cos(0,4) {float(unit(E[0]) @ unit(E[4])):.3f}, "
              f"{time.time() - t0:.0f}s, peak vram {torch.cuda.max_memory_reserved() / 2**30:.2f}G", flush=True)
        torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", default=rd.DEFAULT_SHARE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--embedders", default=",".join(EMBEDDERS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    out = os.path.abspath(a.out)
    tab = os.path.join(out, "tables")
    os.makedirs(tab, exist_ok=True)
    keys = [k.strip() for k in a.embedders.split(",") if k.strip()]

    print("loading rounds ...", flush=True)
    D = rd.Data(a.share, GTE_SEED_CACHE)
    leak = np.array([bool(chk.LEAK.search(t)) for t in D.text])
    degen = D.degenerate.astype(bool)
    keeps = {"all": np.ones(len(D.r), bool), "clean": ~leak & ~degen}
    rng = np.random.default_rng(SEED)
    P = int(D.rounds.patient.max()) + 1
    pats = np.unique(D.rounds.patient)
    splits = []
    for _ in range(N_SPLITS):
        p = rng.permutation(pats)
        A, B = np.zeros(P, bool), np.zeros(P, bool)
        A[p[: len(p) // 2]], B[p[len(p) // 2:]] = True, True
        splits.append((A, B))
    boot = np.zeros((N_BOOT, P))                     # patient multiplicities, shared by every cell
    for b in range(N_BOOT):
        np.add.at(boot[b], rng.choice(pats, size=len(pats), replace=True), 1.0)

    pool, n_distinct = chk.sentence_pool(D.text)
    pool.to_csv(os.path.join(tab, "sentence_pool.csv"), index=False)
    cpool = pool[~pool.has_marker].reset_index(drop=True)
    S = cpool.sentence.to_numpy()
    cat_masks = {k: np.array([bool(rx.search(s)) for s in S]) for k, rx in CATEGORIES.items()}
    pd.DataFrame([dict(category=k, n_sentences=int(m.sum()), examples=" | ".join(S[m][:6])) for k, m in cat_masks.items()]) \
        .to_csv(os.path.join(tab, "category_sizes.csv"), index=False)
    print(f"  pool {len(pool):,} ({n_distinct:,} distinct 4-10-word sentences); clean pool {len(cpool):,}; "
          + ", ".join(f"{k} {int(m.sum())}" for k, m in cat_masks.items()), flush=True)

    from wordfreq import top_n_list
    general = np.array([w for w in top_n_list("en", 40000) if re.fullmatch(r"[a-z]{3,}", w)][:20000])
    df_ = collections.Counter()
    for t, k in zip(D.text, keeps["clean"]):
        if k:
            df_.update(set(w for w in _WORDTOK.findall(t.lower()) if re.fullmatch(r"[a-z]{3,}", w)))
    corpus = np.array(sorted(w for w, c in df_.items() if c >= 100))
    print(f"  vocabularies: general {len(general):,}, corpus {len(corpus):,}", flush=True)

    print("lexical log-odds (no embedder) ...", flush=True)
    lex = lexical_logodds(D, keeps["clean"])
    lex.round(3).to_csv(os.path.join(tab, "lexical_logodds.csv"), index=False)
    lines = ["# Words that win within a round, no embedder (Monroe et al. 2008 log-odds, best vs worst, clean)\n",
             "| arm | iterations | win (z) | lose (z) |", "|---|---|---|---|"]
    for (arm, it), g in lex.groupby(["arm", "iteration"], sort=False):
        w = g.sort_values("z", ascending=False)
        lines.append(f"| {LAB[arm]} | {it} | " + ", ".join(f"{r.word} ({r.z:.1f})" for r in w.head(10).itertuples())
                     + " | " + ", ".join(f"{r.word} ({r.z:.1f})" for r in w.tail(10)[::-1].itertuples()) + " |")
    open(os.path.join(tab, "lexical_logodds.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    cos_rows, cat_rows, sent_rows, proj = [], [], [], {}
    for key in keys:
        label = EMBEDDERS[key][0]
        print(f"\n== {label} ==", flush=True)
        t0 = time.time()
        E = embed_cached(key, D.text, out)
        Ep = embed_cached(key, S.tolist(), out)
        Ewg = embed_cached(key, general.tolist(), out)
        Ewc = embed_cached(key, corpus.tolist(), out)
        print(f"  embeddings ready ({time.time() - t0:.0f}s); dim {E.shape[1]}", flush=True)
        word_md = []
        for variant in ("all", "clean"):
            for est in ("winlose", "advw"):
                V, valid, _ = round_vectors(D, E, keeps[variant], est)
                C = Cells(D, V, valid, splits)
                cos_rows += cosine_rows(C, key, variant, est)
                cat_rows += category_rows(C, Ep, cat_masks, boot, key, variant, est)
                if variant == "clean" and est == "winlose":
                    txt, rows = sentence_tables(C, Ep, S, key, label)
                    open(os.path.join(tab, f"top_sentences_{key}.md"), "w", encoding="utf-8").write(txt)
                    sent_rows += rows
                    word_md.append(word_tables(C, Ewg, general, key, label, "general (20k)"))
                    word_md.append(word_tables(C, Ewc, corpus, key, label, "corpus"))
                    proj[key] = {c: Ep @ C.full[c] for c in CELLS}
                    proj[key].update({("K0-K5", it): Ep @ (C.full[("GRPO_K0", it)] - C.full[("GRPO_K5", it)])
                                      for it in ITERS})
                del V
        open(os.path.join(tab, f"top_words_{key}.md"), "w", encoding="utf-8").write(
            f"# {label}: top words per direction\n" + "".join(word_md))
        pd.DataFrame(cos_rows).round(4).to_csv(os.path.join(tab, "cosines.csv"), index=False)
        pd.DataFrame(cat_rows).round(4).to_csv(os.path.join(tab, "categories.csv"), index=False)
        pd.DataFrame(sent_rows).to_csv(os.path.join(tab, "top_sentences.csv"), index=False)
        del E, Ep, Ewg, Ewc
        gc.collect()
        print(f"  {label} done in {time.time() - t0:.0f}s", flush=True)

    cos, cat = pd.DataFrame(cos_rows), pd.DataFrame(cat_rows)
    agree = []
    ks = list(proj)
    for c in list(proj[ks[0]]):
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                agree.append(dict(cell=f"{c[0]}|{c[1]}", a=ks[i], b=ks[j],
                                  spearman=float(spearmanr(proj[ks[i]][c], proj[ks[j]][c])[0])))
    pd.DataFrame(agree).round(4).to_csv(os.path.join(tab, "embedder_agreement.csv"), index=False)
    figures(cos, cat, out)
    summary = dict(rounds=int(len(D.rounds)), candidates=int(len(D.r)), leaked=float(leak.mean()),
                   degenerate=float(degen.mean()), pool=int(len(pool)), clean_pool=int(len(cpool)),
                   categories={k: int(m.sum()) for k, m in cat_masks.items()},
                   vocab_general=int(len(general)), vocab_corpus=int(len(corpus)), embedders=keys)
    json.dump(summary, open(os.path.join(tab, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("done:", out)


if __name__ == "__main__":
    main()
