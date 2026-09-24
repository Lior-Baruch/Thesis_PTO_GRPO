"""
process.py — utterance-level MI PROCESS analysis (family ``lookahead/process``).

Every MI instrument in the lake is a per-conversation aggregate: MITI hands back seven behaviour
COUNTS, PCT three change-talk COUNTS, MICI six MI-inconsistent COUNTS. None says *where* in the
session a behaviour happened or *what the patient said next*. The ``MIPROC`` instrument
(``code/questionnaires.py``, one code per therapist utterance + one per patient utterance, in
order) makes that visible, and this module turns the two code strings into the classic MI
process-research quantities (Moyers & Martin 2006 style sequential analysis):

- **yield** — P(next patient utterance is change talk | therapist code): does a reflection or an
  open question *produce* change talk more often than praise or persuasion does, and does
  look-ahead change those yields?
- **responsiveness** — P(therapist code | preceding patient code): what does the policy do after
  sustain talk vs change talk (reflect change talk? praise sustain talk?).
- **within-session change-talk trajectory** and **time to first change talk**.
- **code-sequence habits** — question chains (interrogation), reflection-after-change-talk,
  praise-after-sustain-talk.
- **parity** — the per-utterance counts summed per conversation vs the conversation-level MITI and
  PCT counts the same grader already produced: a free validity check of the new coder.

Loaders read the lake under the ACTIVE judge (``constants.set_active_judge``), so the notebook
loops graders and puts them side by side — never averaged. The scripted therapist opener
(therapist #1) is excluded from every rate and every transition: it is the prompt, not the policy.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import BOOT_SEED, PERSONA_COLS
from .data import iter_conv_rows, load_cached, eval_input_roots

TH_CODES = ["OQ", "CQ", "SR", "CR", "AF", "PRA", "GI", "PERS", "SEEK", "CONF", "OTH"]
PT_CODES = ["CT", "ST", "NEU"]
REFLECT = {"SR", "CR"}
QUESTION = {"OQ", "CQ"}
MI_INCONSISTENT = {"PRA", "PERS", "CONF"}
MI_ADHERENT = {"AF", "SEEK", "SR", "CR", "OQ"}

_STATE = ["arm", "method", "K", "model", "iteration", "is_base"]
_CONV = _STATE + ["file_index"]

#: Per-conversation metrics that get the persona-paired K contrast.
PROCESS_K_METRICS = ([f"th_{c}_rate" for c in TH_CODES] +
                     ["pct_oq", "pct_cr", "rtoq", "mi_incons_rate", "mi_adherent_rate",
                      "ct_prop", "st_prop", "reached_ct", "first_ct_pos",
                      "ct_after_q", "ct_after_refl", "ct_after_pra", "refl_after_ct", "pra_after_st",
                      "pers_after_st", "refl_after_st",
                      "q_chain_rate", "ct_late_minus_early"])
PROCESS_METRIC_LABELS = {
    **{f"th_{c}_rate": f"therapist {c} share of turns" for c in TH_CODES},
    "pct_oq": "open share of questions OQ/(OQ+CQ)", "pct_cr": "complex share of reflections CR/(SR+CR)",
    "rtoq": "reflections per question (SR+CR)/(OQ+CQ)",
    "mi_incons_rate": "MI-inconsistent share of turns (PRA+PERS+CONF)",
    "mi_adherent_rate": "MI-adherent share of turns (OQ+SR+CR+AF+SEEK)",
    "ct_prop": "change-talk share of patient turns", "st_prop": "sustain-talk share of patient turns",
    "reached_ct": "conversation reached any change talk (0/1)",
    "first_ct_pos": "patient turn index of the first change talk (reached conversations only)",
    "ct_after_q": "P(change talk | preceding therapist question)",
    "ct_after_refl": "P(change talk | preceding therapist reflection)",
    "ct_after_pra": "P(change talk | preceding non-specific praise)",
    "refl_after_ct": "P(therapist reflects | preceding patient change talk)",
    "pra_after_st": "P(therapist praises | preceding patient sustain talk)",
    # 2026-09-24: the other two answers to sustain talk -- MI's (reflect it) and the righting
    # reflex (argue for change) -- so the praise row cannot be read on its own.
    "pers_after_st": "P(therapist persuades | preceding patient sustain talk)",
    "refl_after_st": "P(therapist reflects | preceding patient sustain talk)",
    "q_chain_rate": "share of questions that follow another question",
    "ct_late_minus_early": "change-talk share, second half − first half of the session",
}
LOWER_BETTER = {"th_PRA_rate", "th_PERS_rate", "th_CONF_rate", "th_CQ_rate", "mi_incons_rate", "st_prop",
                "first_ct_pos", "pra_after_st", "pers_after_st", "q_chain_rate"}
PROCESS_FAMILIES = {
    "therapist_codes": [f"th_{c}_rate" for c in TH_CODES],
    "technique": ["pct_oq", "pct_cr", "rtoq", "mi_incons_rate", "mi_adherent_rate", "q_chain_rate"],
    "patient": ["ct_prop", "st_prop", "reached_ct", "first_ct_pos", "ct_late_minus_early"],
    "contingency": ["ct_after_q", "ct_after_refl", "ct_after_pra", "refl_after_ct", "pra_after_st",
                    "pers_after_st", "refl_after_st"],
}
#: Patient-turn bins for the within-session change-talk trajectory (patient turn index, 0-based).
CT_BINS = [(0, 1, "1-2"), (2, 4, "3-5"), (5, 8, "6-9"), (9, 10 ** 6, "10+")]


def _arms(arms):
    from . import discover_arms
    return discover_arms() if arms is None else arms


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Loaders                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def load_miproc(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Per (arm, iteration, conversation) MIPROC row: the two code strings + every derived metric
    of :data:`PROCESS_K_METRICS`, recomputed here from the codes with the opener excluded.
    Read under the ACTIVE judge; parquet-cached per judge."""
    arms = _arms(arms)
    return load_cached("miproc", arms, lambda: _load_miproc_impl(arms, attach_persona=attach_persona),
                       input_roots=eval_input_roots(arms),
                       params={"attach_persona": attach_persona, "metrics": tuple(PROCESS_K_METRICS), "v": 1})


def _load_miproc_impl(arms, *, attach_persona: bool) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for k in arm.iters:
            for fi, r in iter_conv_rows(arm.eval_dir(k, "MIPROC")):
                th = str(r.get("MIPROC_ThCodes", "-")); pt = str(r.get("MIPROC_PtCodes", "-"))
                th_codes = [] if th in ("-", "nan", "") else th.split("|")
                pt_codes = [] if pt in ("-", "nan", "") else pt.split("|")
                row = {"arm": arm.label, "method": arm.method, "K": arm.K, "model": arm.model_name(k),
                       "iteration": k, "is_base": (k == 0), "file_index": fi,
                       "th_codes": "|".join(th_codes), "pt_codes": "|".join(pt_codes),
                       "n_th": len(th_codes), "n_pt": len(pt_codes)}
                row.update(conversation_metrics(th_codes, pt_codes))
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        from .behavior import _attach_by_arm
        df = _attach_by_arm(df, arms)
    return df


def conversation_metrics(th_codes: Sequence[str], pt_codes: Sequence[str]) -> dict:
    """Every per-conversation process metric from the two code sequences.

    Alternation is strict, therapist first: therapist #i (0-based) is answered by patient #i, and
    therapist #i (i ≥ 1) answers patient #(i−1). Therapist #0 is the scripted opener and is
    excluded from every rate and every transition; patient #0 (the reply to the opener) stays in
    the patient-side quantities.
    """
    th = list(th_codes); pt = list(pt_codes)
    body = th[1:]                       # policy turns
    n = len(body)
    out = {}
    c = Counter(body)
    for code in TH_CODES:
        out[f"th_{code}_rate"] = c.get(code, 0) / n if n else np.nan
    q = c.get("OQ", 0) + c.get("CQ", 0); refl = c.get("SR", 0) + c.get("CR", 0)
    out["pct_oq"] = c.get("OQ", 0) / q if q else np.nan
    out["pct_cr"] = c.get("CR", 0) / refl if refl else np.nan
    out["rtoq"] = refl / q if q else np.nan
    out["mi_incons_rate"] = sum(c.get(x, 0) for x in MI_INCONSISTENT) / n if n else np.nan
    out["mi_adherent_rate"] = sum(c.get(x, 0) for x in MI_ADHERENT) / n if n else np.nan
    pc = Counter(pt); m = len(pt)
    out["ct_prop"] = pc.get("CT", 0) / m if m else np.nan
    out["st_prop"] = pc.get("ST", 0) / m if m else np.nan
    first = next((i for i, x in enumerate(pt) if x == "CT"), None)
    out["reached_ct"] = float(first is not None) if m else np.nan
    out["first_ct_pos"] = float(first) if first is not None else np.nan
    # contingencies over (therapist #i, patient #i) pairs, i ≥ 1
    pairs = [(th[i], pt[i]) for i in range(1, min(len(th), len(pt)))]
    def _p_ct(pred):
        sel = [p for t, p in pairs if pred(t)]
        return sum(p == "CT" for p in sel) / len(sel) if sel else np.nan
    out["ct_after_q"] = _p_ct(lambda t: t in QUESTION)
    out["ct_after_refl"] = _p_ct(lambda t: t in REFLECT)
    out["ct_after_pra"] = _p_ct(lambda t: t == "PRA")
    # responsiveness over (patient #(i−1), therapist #i) pairs, i ≥ 1
    resp = [(pt[i - 1], th[i]) for i in range(1, min(len(th), len(pt) + 1))]
    after_ct = [t for p, t in resp if p == "CT"]; after_st = [t for p, t in resp if p == "ST"]
    out["refl_after_ct"] = sum(t in REFLECT for t in after_ct) / len(after_ct) if after_ct else np.nan
    out["pra_after_st"] = sum(t == "PRA" for t in after_st) / len(after_st) if after_st else np.nan
    out["pers_after_st"] = sum(t == "PERS" for t in after_st) / len(after_st) if after_st else np.nan
    out["refl_after_st"] = sum(t in REFLECT for t in after_st) / len(after_st) if after_st else np.nan
    qs = [i for i, t in enumerate(body) if t in QUESTION]
    out["q_chain_rate"] = (sum(1 for i in qs if i > 0 and body[i - 1] in QUESTION) / len(qs)) if qs else np.nan
    if m >= 4:
        half = m // 2
        out["ct_late_minus_early"] = (sum(x == "CT" for x in pt[half:]) / (m - half)) - (sum(x == "CT" for x in pt[:half]) / half)
    else:
        out["ct_late_minus_early"] = np.nan
    return out


def utterance_long(conv: pd.DataFrame) -> pd.DataFrame:
    """Explode the per-conversation code strings: one row per utterance with ``role``, ``position``
    (0-based within the role; therapist 0 = the opener) and ``code``."""
    rows = []
    keep = [c for c in _CONV + ["persona_id"] if c in conv.columns]
    for _, r in conv.iterrows():
        base = {c: r[c] for c in keep}
        for role, col in (("therapist", "th_codes"), ("patient", "pt_codes")):
            codes = str(r[col]).split("|") if isinstance(r[col], str) and r[col] else []
            for i, code in enumerate(codes):
                rows.append({**base, "role": role, "position": i, "code": code})
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Sequential analysis                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _pairs(conv: pd.DataFrame):
    """Yield (state key, therapist code, patient reply code, previous patient code) per policy turn."""
    for _, r in conv.iterrows():
        th = r["th_codes"].split("|") if r["th_codes"] else []
        pt = r["pt_codes"].split("|") if r["pt_codes"] else []
        key = tuple(r[c] for c in _STATE)
        for i in range(1, len(th)):
            reply = pt[i] if i < len(pt) else None
            prev = pt[i - 1] if i - 1 < len(pt) else None
            yield key, th[i], reply, prev


def transition_yield(conv: pd.DataFrame) -> pd.DataFrame:
    """Per (state, therapist code): n turns, and P(next patient = CT / ST / NEU) — the YIELD table.
    Wilson 95 % interval on the CT yield."""
    counts: Dict[tuple, Counter] = {}
    for key, t, reply, _ in _pairs(conv):
        if reply is None:
            continue
        counts.setdefault((key, t), Counter())[reply] += 1
    rows = []
    for (key, t), c in counts.items():
        n = sum(c.values())
        p = c.get("CT", 0) / n
        z = 1.96; den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den; half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        row = dict(zip(_STATE, key)); row.update({"th_code": t, "n": n, "p_ct": p, "p_ct_lo": centre - half,
                                                  "p_ct_hi": centre + half, "p_st": c.get("ST", 0) / n,
                                                  "p_neu": c.get("NEU", 0) / n})
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(TH_CODES)}
    return out.assign(_o=out["th_code"].map(order)).sort_values(["arm", "iteration", "_o"]).drop(columns="_o").reset_index(drop=True)


def responsiveness(conv: pd.DataFrame) -> pd.DataFrame:
    """Per (state, preceding patient code): distribution of the therapist's next code — what the
    policy does after change talk vs sustain talk (share per therapist code)."""
    counts: Dict[tuple, Counter] = {}
    for key, t, _, prev in _pairs(conv):
        if prev is None:
            continue
        counts.setdefault((key, prev), Counter())[t] += 1
    rows = []
    for (key, prev), c in counts.items():
        n = sum(c.values())
        row = dict(zip(_STATE, key)); row.update({"pt_code": prev, "n": n})
        row.update({f"p_{t}": c.get(t, 0) / n for t in TH_CODES})
        row["p_reflect"] = sum(c.get(t, 0) for t in REFLECT) / n
        row["p_question"] = sum(c.get(t, 0) for t in QUESTION) / n
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "iteration", "pt_code"]).reset_index(drop=True)


def transition_matrix(conv: pd.DataFrame, arm: str, iteration: int) -> pd.DataFrame:
    """Therapist code × next patient code contingency table (row-normalised) for one state."""
    sub = conv[(conv["arm"] == arm) & (conv["iteration"] == iteration)]
    c = Counter()
    for _, t, reply, _ in _pairs(sub):
        if reply is not None:
            c[(t, reply)] += 1
    M = pd.DataFrame(0.0, index=TH_CODES, columns=PT_CODES)
    for (t, p), v in c.items():
        M.loc[t, p] = v
    n = M.sum(axis=1)
    return M.div(n.replace(0, np.nan), axis=0).assign(n=n.astype(int))


def _bin_of(pos: int) -> str:
    for lo, hi, lab in CT_BINS:
        if lo <= pos <= hi:
            return lab
    return "?"


def ct_trajectory(conv: pd.DataFrame) -> pd.DataFrame:
    """Per (state, patient turn bin): change-talk and sustain-talk share, n utterances, share of
    the state's conversations reaching the bin."""
    long = utterance_long(conv)
    pt = long[long["role"] == "patient"].copy()
    pt["bin"] = pt["position"].map(_bin_of)
    n_conv = pt.groupby(_STATE, sort=False)["file_index"].nunique().rename("n_convs_state")
    agg = (pt.assign(ct=(pt["code"] == "CT").astype(float), st=(pt["code"] == "ST").astype(float))
           .groupby(_STATE + ["bin"], sort=False)
           .agg(ct_prop=("ct", "mean"), st_prop=("st", "mean"), n=("code", "size"),
                n_convs=("file_index", "nunique")).reset_index())
    agg = agg.merge(n_conv.reset_index(), on=_STATE)
    agg["share_convs_reaching"] = agg["n_convs"] / agg["n_convs_state"]
    order = {lab: i for i, (_, _, lab) in enumerate(CT_BINS)}
    return (agg.assign(_o=agg["bin"].map(order)).sort_values(["arm", "iteration", "_o"])
            .drop(columns="_o").reset_index(drop=True))


# ── Persistence: does change talk, once voiced, stay? ───────────────────────────────────────────
# The yield table conditions on the THERAPIST's code only. The patient's code is strongly
# autocorrelated (a change-talk turn is usually followed by another), so a code the policy places
# right after change talk inherits a high yield whatever it does. Conditioning on the patient's
# PREVIOUS code separates the two: P(reply = CT | previous patient code) is the persistence of
# change talk (and the conversion rate out of sustain talk), and the same split per therapist code
# says whether a code adds anything over the patient's own momentum.

def _wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def persistence_by_state(conv: pd.DataFrame) -> pd.DataFrame:
    """Per (state, previous patient code): the distribution of the patient's NEXT code, pooled over
    turns. ``prev_code == "CT"`` rows give the persistence of change talk (``p_ct``) and its relapse
    to sustain talk (``p_st``); ``prev_code == "ST"`` rows give the conversion out of sustain talk.
    Each pair is (patient #(i−1), patient #i) with therapist #i between them, i ≥ 1 — the opener's
    reply is a ``prev``, never a ``reply``. Wilson 95% interval on ``p_ct``."""
    counts: Dict[tuple, Counter] = {}
    for key, _, reply, prev in _pairs(conv):
        if reply is None or prev is None:
            continue
        counts.setdefault((key, prev), Counter())[reply] += 1
    rows = []
    for (key, prev), c in counts.items():
        n = sum(c.values())
        lo, hi = _wilson(c.get("CT", 0), n)
        row = dict(zip(_STATE, key))
        row.update({"prev_code": prev, "n": n, "p_ct": c.get("CT", 0) / n, "p_ct_lo": lo, "p_ct_hi": hi,
                    "p_st": c.get("ST", 0) / n, "p_neu": c.get("NEU", 0) / n})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["arm", "iteration", "prev_code"]).reset_index(drop=True)


#: Per-conversation persistence metrics (NaN where the conversation has no such previous code).
PERSISTENCE_METRICS = ["ct_persist", "ct_relapse", "st_to_ct"]
PERSISTENCE_LABELS = {
    "ct_persist": "P(patient change talk | previous patient turn was change talk)",
    "ct_relapse": "P(patient sustain talk | previous patient turn was change talk)",
    "st_to_ct": "P(patient change talk | previous patient turn was sustain talk)",
}


def persistence_metrics(conv: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation: :data:`PERSISTENCE_METRICS`, computed within the conversation
    over the same (patient #(i−1), patient #i) pairs as :func:`persistence_by_state` — the unit the
    persona-paired K contrast needs. Carries the conversation keys and ``persona_id`` (when present)."""
    keep = [c for c in _CONV + ["persona_id"] + list(PERSONA_COLS) if c in conv.columns]
    rows = []
    for _, r in conv.iterrows():
        th = r["th_codes"].split("|") if r["th_codes"] else []
        pt = r["pt_codes"].split("|") if r["pt_codes"] else []
        pairs = [(pt[i - 1], pt[i]) for i in range(1, min(len(th), len(pt)))]
        after_ct = [b for a, b in pairs if a == "CT"]
        after_st = [b for a, b in pairs if a == "ST"]
        row = {c: r[c] for c in keep}
        row["ct_persist"] = sum(b == "CT" for b in after_ct) / len(after_ct) if after_ct else np.nan
        row["ct_relapse"] = sum(b == "ST" for b in after_ct) / len(after_ct) if after_ct else np.nan
        row["st_to_ct"] = sum(b == "CT" for b in after_st) / len(after_st) if after_st else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def conditioned_yield(conv: pd.DataFrame) -> pd.DataFrame:
    """Per (state, previous patient code, therapist code): n and P(next patient = CT), Wilson 95%.

    The yield of :func:`transition_yield` split by what the patient said BEFORE the therapist's
    turn. A code whose yield after change talk matches every other code's is riding the patient's
    momentum; one that lifts the yield after sustain talk is doing something."""
    counts: Dict[tuple, Counter] = {}
    for key, t, reply, prev in _pairs(conv):
        if reply is None or prev is None:
            continue
        counts.setdefault((key, prev, t), Counter())[reply] += 1
        counts.setdefault((key, prev, "ALL"), Counter())[reply] += 1
    rows = []
    for (key, prev, t), c in counts.items():
        n = sum(c.values())
        lo, hi = _wilson(c.get("CT", 0), n)
        row = dict(zip(_STATE, key))
        row.update({"prev_code": prev, "th_code": t, "n": n, "p_ct": c.get("CT", 0) / n,
                    "p_ct_lo": lo, "p_ct_hi": hi})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    order = {c: i for i, c in enumerate(TH_CODES + ["ALL"])}
    out = pd.DataFrame(rows)
    return (out.assign(_o=out["th_code"].map(order)).sort_values(["arm", "iteration", "prev_code", "_o"])
            .drop(columns="_o").reset_index(drop=True))


def state_table(conv: pd.DataFrame, metrics: Sequence[str] = PROCESS_K_METRICS) -> pd.DataFrame:
    g = conv.groupby(_STATE, sort=False)
    out = g[list(metrics)].mean(); se = g[list(metrics)].sem().add_suffix("_se")
    n = g.size().rename("n")
    return pd.concat([out, se, n], axis=1).reset_index().sort_values(["arm", "iteration"]).reset_index(drop=True)


def to_scores_long(conv: pd.DataFrame, arms, metrics: Sequence[str] = PROCESS_K_METRICS) -> pd.DataFrame:
    """The ``load_scores_long`` schema (questionnaire = metric id) so every ``stats`` / ``lookahead``
    entry point applies unchanged."""
    from .behavior import _attach_by_arm
    df = conv if "persona_id" in conv.columns else _attach_by_arm(conv, arms)
    df = df.assign(oracle=df["arm"].map({a.label: a.oracle for a in arms}))
    id_cols = ["method", "arm", "K", "oracle", "model", "iteration", "is_base", "file_index", "persona_id"] + \
              [c for c in PERSONA_COLS if c in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=[m for m in metrics if m in df.columns],
                   var_name="questionnaire", value_name="score")
    return long.dropna(subset=["score"]).reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Parity with the conversation-level instruments                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
_MITI_MAP = {"B3_Q": ("OQ", "CQ"), "B4_SR": ("SR",), "B5_CR": ("CR",), "B6_AF": ("AF",),
             "B1_GI": ("GI",), "B2_Persuade": ("PERS",), "B7_Seek": ("SEEK",)}
_PCT_MAP = {"PCT_ChangeTalk": "CT", "PCT_SustainTalk": "ST", "PCT_Neutral": "NEU"}


def parity(conv: pd.DataFrame, miti: pd.DataFrame, pct: pd.DataFrame) -> pd.DataFrame:
    """MIPROC per-utterance counts summed per conversation vs the same grader's conversation-level
    MITI / PCT counts: per (arm, iteration, instrument count) Spearman ρ across conversations, mean
    absolute difference, and the two means. Counts here INCLUDE the opener (MITI's do)."""
    from scipy.stats import spearmanr
    keys = ["arm", "iteration", "file_index"]
    rows = []
    c = conv.copy()
    for col, codes in _MITI_MAP.items():
        c[f"mp_{col}"] = c["th_codes"].map(lambda s: sum(1 for x in s.split("|") if x in codes) if s else 0)
    for col, code in _PCT_MAP.items():
        c[f"mp_{col}"] = c["pt_codes"].map(lambda s: sum(1 for x in s.split("|") if x == code) if s else 0)
    frames = []
    if miti is not None and not miti.empty:
        frames.append(("MITI", c.merge(miti[keys + list(_MITI_MAP)], on=keys, how="inner"), list(_MITI_MAP)))
    if pct is not None and not pct.empty:
        frames.append(("PCT", c.merge(pct[keys + list(_PCT_MAP)], on=keys, how="inner"), list(_PCT_MAP)))
    for inst, j, cols in frames:
        for key, g in j.groupby(_STATE, sort=False):
            for col in cols:
                a, b = g[f"mp_{col}"].astype(float), g[col].astype(float)
                ok = a.notna() & b.notna()
                if ok.sum() < 10:
                    continue
                rho = spearmanr(a[ok], b[ok]).correlation if (a[ok].nunique() > 1 and b[ok].nunique() > 1) else np.nan
                row = dict(zip(_STATE, key))
                row.update({"instrument": inst, "count": col, "n": int(ok.sum()), "rho": float(rho),
                            "mean_miproc": float(a[ok].mean()), "mean_instrument": float(b[ok].mean()),
                            "mean_abs_diff": float((a[ok] - b[ok]).abs().mean())})
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["instrument", "count", "arm", "iteration"]).reset_index(drop=True)


def parity_pooled(par: pd.DataFrame) -> pd.DataFrame:
    """Fisher-z pooled ρ and mean |Δ| per (instrument, count), n-weighted over states."""
    rows = []
    for (inst, col), g in par.groupby(["instrument", "count"]):
        z = np.arctanh(np.clip(g["rho"].to_numpy(float), -0.999, 0.999)); w = g["n"].to_numpy(float)
        ok = np.isfinite(z)
        rows.append({"instrument": inst, "count": col, "n_states": int(len(g)),
                     "rho_pooled": float(np.tanh((z[ok] * w[ok]).sum() / w[ok].sum())) if ok.any() else np.nan,
                     "mean_abs_diff": float((g["mean_abs_diff"] * w).sum() / w.sum()),
                     "mean_miproc": float((g["mean_miproc"] * w).sum() / w.sum()),
                     "mean_instrument": float((g["mean_instrument"] * w).sum() / w.sum())})
    return pd.DataFrame(rows)


def process_numbers(levels_by_judge: Dict[str, pd.DataFrame], yields_by_judge: Dict[str, pd.DataFrame],
                    parity_by_judge: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    out = {}
    for j, lv in levels_by_judge.items():
        for a, g in lv.groupby("arm"):
            g = g.sort_values("iteration")
            for m in ("ct_prop", "mi_incons_rate", "th_PRA_rate", "th_OQ_rate", "rtoq", "ct_after_q", "refl_after_ct", "reached_ct"):
                if m in g.columns:
                    out[f"levels.{j}.{a}.{m}.base"] = {"value": round(float(g.iloc[0][m]), 4), "source": "process_levels", "note": "iteration 0"}
                    out[f"levels.{j}.{a}.{m}.final"] = {"value": round(float(g.iloc[-1][m]), 4), "source": "process_levels",
                                                        "note": f"iteration {int(g.iloc[-1]['iteration'])}"}
    for j, y in yields_by_judge.items():
        for a, g in y.groupby("arm"):
            last = g[g["iteration"] == g["iteration"].max()]
            for _, r in last.iterrows():
                out[f"yield.{j}.{a}.{r['th_code']}.p_ct.final"] = {"value": round(float(r["p_ct"]), 4), "source": "transition_yield",
                                                                  "note": f"n = {int(r['n'])} turns, iteration {int(r['iteration'])}"}
    for j, p in parity_by_judge.items():
        for _, r in p.iterrows():
            out[f"parity.{j}.{r['instrument']}.{r['count']}.rho_pooled"] = {"value": round(float(r["rho_pooled"]), 4),
                                                                            "source": "parity_pooled", "note": "Fisher-z pooled within-state ρ"}
    return out
