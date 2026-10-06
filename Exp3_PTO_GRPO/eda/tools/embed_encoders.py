"""embed_encoders.py — the GPU step behind ``eda_analysis.encoders``: fill the per-encoder caches.

WHAT
----
For each encoder key of ``eda_analysis.encoders.ENCODERS`` (minilm, gte, mxbai, qwen3, llama8), make
sure every text the multi-encoder direction readouts need has an embedding in

    eda/.emb_cache/encoders/<key>/{texts.json.gz, emb.npy}      (gitignored: **/.emb_cache/)

``texts.json.gz`` is the list of texts (keyed by the exact string), ``emb.npy`` the unit-normalised
rows in the same order. The texts are ``encoders.required_texts`` of the two GRPO runs' training
candidates — exactly what ``pref.load_weighted_candidates(<GRPO arms>, drop_zero_weight=False)``
yields (train phase, every candidate of every gradient group, empty completions excluded) — plus
the sentence pool ``encoders.sentence_pool`` builds from them.

WHERE THE VECTORS COME FROM (in this order; the cache is append-only)
------------------------------------------------------------------
1. ``minilm`` only: the EDA's own MiniLM cache, ``eda/.emb_cache/all-MiniLM-L6-v2.pkl`` (sha1-keyed,
   float32) — the vectors behind ``direction_k_by_iter_grpo`` / ``direction_stability_grpo``, so the
   minilm rows of the multi-encoder tables reproduce those tables exactly. Stored float32.
2. The 2026-10-06 meetings cache, ``meetings/2026-10-06_embedding_directions/.emb_cache/<key>/``
   (``--seed-dir``; same layout, float16) — the run this port comes from.
3. Anything still missing is embedded on the GPU with the meetings script's encoders (same models,
   pooling, BOS exclusion). Stored float16, except minilm (float32, run in float32 as
   ``pref._embed_texts`` does).

VRAM — an over-budget request REBOOTS the local 12 GB card's PC instead of raising, so every model
runs under ``torch.cuda.set_per_process_memory_fraction(0.5)`` (an overrun then raises
OutOfMemoryError) with length-sorted, bounded batches; batch sizes are smaller than the meetings
script's. ``--dry-run`` loads no model and imports no torch.

USAGE (from Exp3_PTO_GRPO/eda, with the repo venv)
-----
    ..\\..\\.venv\\Scripts\\python.exe tools/embed_encoders.py --dry-run          # counts only
    ..\\..\\.venv\\Scripts\\python.exe tools/embed_encoders.py                    # seed + embed what is missing
    ..\\..\\.venv\\Scripts\\python.exe tools/embed_encoders.py --keys gte,llama8  # a subset
    ..\\..\\.venv\\Scripts\\python.exe tools/embed_encoders.py --max-new 5000     # refuse to embed more than this

The working directory does not matter. No API calls.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EDA = os.path.dirname(_HERE)
if _EDA not in sys.path:
    sys.path.insert(0, _EDA)

from eda_analysis.constants import WORKSPACE_ROOT                   # noqa: E402
from eda_analysis import encoders as enc                            # noqa: E402
from eda_analysis import pref                                       # noqa: E402
from eda_analysis.data import discover_arms                         # noqa: E402

_REPO = os.path.dirname(WORKSPACE_ROOT)
DEFAULT_SEED_DIR = os.path.join(_REPO, "meetings", "2026-10-06_embedding_directions", ".emb_cache")
EDA_MINILM_PKL = os.path.join(pref._CACHE_DIR, "all-MiniLM-L6-v2.pkl")
GRPO_ARMS = ("GRPO_LA0", "GRPO_LA5")
LLAMA = "meta-llama/Llama-3.2-1B"
GTE = "Alibaba-NLP/gte-base-en-v1.5"

# key: (kind, model id, torch dtype name, max tokens, batch, stored dtype). Copied from the meetings
# script's EMBEDDERS, batches halved; minilm runs in float32 like pref._embed_texts.
SPEC = {
    "minilm": ("st", "sentence-transformers/all-MiniLM-L6-v2", "float32", 256, 64, np.float32),
    "gte": ("gte", GTE, None, 512, None, np.float16),
    "mxbai": ("st", "mixedbread-ai/mxbai-embed-large-v1", "float16", 512, 32, np.float16),
    "qwen3": ("st", "Qwen/Qwen3-Embedding-0.6B", "bfloat16", 512, 32, np.float16),
    "llama8": ("llama", LLAMA, "bfloat16", 512, None, np.float16),
}
assert list(SPEC) == list(enc.ENCODERS), "SPEC and encoders.ENCODERS must list the same keys, in order"


# ── encoders (copied from meetings/build/embedding_directions_2026-10-06.py; torch imported lazily) ──
def _cap_vram():
    import torch
    torch.cuda.set_per_process_memory_fraction(0.5)


def _free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def st_encode(model_id, dtype, max_len, batch):
    def run(texts):
        import torch
        from sentence_transformers import SentenceTransformer
        _cap_vram()
        m = SentenceTransformer(model_id, device="cuda", model_kwargs={"dtype": getattr(torch, dtype)})
        m.max_seq_length = max_len
        E = m.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=len(texts) > 5000)
        del m
        _free()
        return E
    return run


def gte_encode(texts, max_batch=64, token_budget=8192):
    """Mean-pooled gte-base-en-v1.5 (from analyze_grpo_round_directions_2026-09-29.py's ``_encode``)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    _cap_vram()
    tok = AutoTokenizer.from_pretrained(GTE)
    m = AutoModel.from_pretrained(GTE, trust_remote_code=True).eval()
    # transformers 5 leaves this remote code's non-persistent buffers uninitialised; rebuild them.
    m.embeddings.register_buffer("position_ids", torch.arange(m.config.max_position_embeddings), persistent=False)
    m.embeddings._init_rope(m.config)
    m = m.cuda()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 768), dtype=np.float32)
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
    _free()
    return out


def llama_encode(texts, layer=8, max_batch=64, token_budget=8192):
    """Llama-3.2-1B, output of decoder layer ``layer`` (before the final RMSNorm), mean-pooled over
    the text's tokens with BOS EXCLUDED (a mid-layer BOS activation is orders of magnitude larger than
    any other token's and would dominate the mean)."""
    import torch
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
    out = np.zeros((len(texts), m.config.hidden_size), dtype=np.float32)
    i = 0
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
    del m
    _free()
    return out


def encoder_for(key):
    kind, mid, dtype, max_len, batch, _ = SPEC[key]
    if kind == "st":
        return st_encode(mid, dtype, max_len, batch)
    if kind == "gte":
        return gte_encode
    return llama_encode


# ── cache io ──────────────────────────────────────────────────────────────────
def _read(cdir):
    tpath, epath = os.path.join(cdir, "texts.json.gz"), os.path.join(cdir, "emb.npy")
    if not (os.path.exists(tpath) and os.path.exists(epath)):
        return [], None
    with gzip.open(tpath, "rt", encoding="utf-8") as f:
        known = json.load(f)
    return known, np.load(epath, mmap_mode="r")


def _write(cdir, texts, E):
    """Atomic: write both files to .tmp, then rename (an interrupted run never leaves a half cache)."""
    os.makedirs(cdir, exist_ok=True)
    tpath, epath = os.path.join(cdir, "texts.json.gz"), os.path.join(cdir, "emb.npy")
    with open(epath + ".tmp", "wb") as f:
        np.save(f, E)
    with gzip.open(tpath + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    os.replace(epath + ".tmp", epath)
    os.replace(tpath + ".tmp", tpath)


def required() -> list:
    arms = [a for a in discover_arms() if a.label in GRPO_ARMS]
    if sorted(a.label for a in arms) != sorted(GRPO_ARMS):
        sys.exit(f"need both GRPO arms on disk, found {[a.label for a in arms]}")
    cands = pref.load_weighted_candidates(arms, drop_zero_weight=False)
    pool = enc.sentence_pool(cands["completion"].astype(str))
    texts = enc.required_texts(cands, pool)
    print(f"GRPO training candidates: {len(cands):,} rows; sentence pool {len(pool):,} "
          f"(eligible {pool.attrs['n_eligible']:,}, marker-dropped {pool.attrs['n_marker']}); "
          f"{len(texts):,} distinct texts to cover", flush=True)
    return texts


def process(key, texts, seed_dir, *, dry_run, max_new):
    _, _, _, _, _, store = SPEC[key]
    cdir = os.path.join(enc.ENCODER_CACHE_DIR, key)
    known, E = _read(cdir)
    have = set(known)
    missing = [t for t in texts if t not in have]
    report = {"encoder": key, "required": len(texts), "in_eda_cache": len(texts) - len(missing)}
    new_texts, new_rows = [], []

    if key == "minilm" and missing and os.path.exists(EDA_MINILM_PKL):
        with open(EDA_MINILM_PKL, "rb") as f:
            pk = pickle.load(f)
        got = [(t, pk[pref._key(t)]) for t in missing if pref._key(t) in pk]
        del pk
        new_texts += [t for t, _ in got]
        new_rows += [np.asarray(v, dtype=np.float32)[None, :] for _, v in got]
        report["from_eda_minilm_pkl"] = len(got)
        got_set = {t for t, _ in got}
        missing = [t for t in missing if t not in got_set]

    sk, sE = _read(os.path.join(seed_dir, key)) if seed_dir else ([], None)
    if missing and sE is not None:
        spos = {t: i for i, t in enumerate(sk)}
        hit = [t for t in missing if t in spos]
        if hit:
            new_texts += hit
            new_rows.append(np.asarray(sE[[spos[t] for t in hit]], dtype=np.float32))
        report["from_meetings_cache"] = len(hit)
        hit_set = set(hit)
        missing = [t for t in missing if t not in hit_set]
    report["missing"] = len(missing)
    print(f"  {key:7s} required {len(texts):,} | already in EDA cache {report['in_eda_cache']:,} | "
          f"from EDA MiniLM pkl {report.get('from_eda_minilm_pkl', 0):,} | from meetings cache "
          f"{report.get('from_meetings_cache', 0):,} | MISSING (to embed) {len(missing):,}", flush=True)
    if dry_run:
        return report
    if len(missing) > max_new:
        print(f"  {key}: {len(missing):,} texts missing > --max-new {max_new:,}; NOT embedding (the candidate set "
              f"differs from the seed caches' — check before spending GPU time). Seeds are still written.", flush=True)
        missing = []
        report["skipped_embedding"] = True
    if missing:
        t0 = time.time()
        X = encoder_for(key)(missing)
        print(f"  {key}: embedded {len(missing):,} texts in {time.time() - t0:.0f}s", flush=True)
        new_texts += missing
        new_rows.append(np.asarray(X, dtype=np.float32))
    if not new_texts:
        return report
    N = np.vstack(new_rows).astype(store)
    if E is not None:
        store = np.result_type(E.dtype, store)          # never downcast an existing cache
        N = np.vstack([np.asarray(E, dtype=store), N.astype(store)])
    _write(cdir, list(known) + new_texts, N.astype(store))
    print(f"  {key}: cache now {N.shape[0]:,} x {N.shape[1]} ({np.dtype(store).name}) -> {cdir}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", default=",".join(enc.ENCODERS), help="comma-separated encoder keys")
    ap.add_argument("--seed-dir", default=DEFAULT_SEED_DIR, help="the meetings cache root ('' = no seeding)")
    ap.add_argument("--dry-run", action="store_true", help="count missing texts; load no model, write nothing")
    ap.add_argument("--max-new", type=int, default=20000,
                    help="refuse to embed more than this many texts per encoder (default 20,000)")
    a = ap.parse_args(argv)
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]
    bad = [k for k in keys if k not in SPEC]
    if bad:
        sys.exit(f"unknown encoder keys {bad}; known: {list(SPEC)}")
    if a.seed_dir and not os.path.isdir(a.seed_dir):
        print(f"(seed dir {a.seed_dir} not found — no seeding)")
        a.seed_dir = ""
    texts = required()
    rows = [process(k, texts, a.seed_dir, dry_run=a.dry_run, max_new=a.max_new) for k in keys]
    print(("DRY RUN — nothing written. " if a.dry_run else "")
          + "missing per encoder: " + ", ".join(f"{r['encoder']} {r['missing']:,}" for r in rows))


if __name__ == "__main__":
    main()
