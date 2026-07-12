"""
behavior.py — what the therapist actually DOES, and how it drifts across iterations.

Two complementary, cross-validating sources:
1. **Oracle MITI behavior counts** (from ``eval_scores/metric=MITI``): questions (B3_Q),
   simple/complex reflections (B4_SR/B5_CR), affirmations (B6_AF), persuasion (B2_Persuade),
   plus the global empathy/change-talk/partnership ratings, and the R:Q ratio.
2. **Deterministic text metrics** (from the conversations): therapist-turn length, verbatim
   repetition loops (degeneration), questions/turn, conversation length.

The structural text metrics are cheap, deterministic, non-LLM cross-checks: they catch
degeneration loops the MITI counts miss and independently confirm the oracle tally (validated:
B3_Q 6.45→3.84, B6_AF 0.42→1.64, loop% 49→0 over PTO LA0 iters 0→10).

⚠ The two **lexical marker rates** (``lex_affirm_marker_rate`` /
``lex_overpraise_marker_rate``) are brittle keyword regexes kept ONLY as a sanity-check that
*validates the direction of* the oracle's MITI_B6_AF (affirmation) and MICI_OverPraise
(sycophancy) counts — they are NOT primary behavior metrics and are deliberately excluded from
``_BEHAVIOR_METRICS``. Use the oracle-coded ``B6_AF`` and ``MICI_OverPraiseRate`` for the real
affirmation/over-praise story; see notebook ``3_Mechanism`` for the cross-check.
"""

import os
import re
from collections import Counter
from typing import List, Optional

import pandas as pd

from .constants import RE_AFFIRM
from .data import (attach_personas,  # personas merged into data.py
                   iter_conv_rows, load_cached, eval_input_roots, conv_input_roots)

_MITI_COLS = {
    "MITI_B1_GI": "B1_GI", "MITI_B3_Q": "B3_Q", "MITI_B4_SR": "B4_SR", "MITI_B5_CR": "B5_CR",
    "MITI_B6_AF": "B6_AF", "MITI_B7_Seek": "B7_Seek", "MITI_B2_Persuade": "B2_Persuade",
    "MITI4_Empathy": "Empathy", "MITI1_CultivatingChangeTalk": "ChangeTalk",
    "MITI2_SofteningSustainTalk": "SoftenSustain",
    "MITI3_Partnership": "Partnership", "MITI_GlobalMean": "MITI_Global",
}

# Lexical marker cues (case-insensitive), matched per therapist turn. These are a
# DIRECTIONAL sanity-check on the oracle's affirmation / over-praise counts, NOT primary
# metrics — see the module docstring. The affirmation cue is shared with pref.py via constants.
_RE_AFFIRM = RE_AFFIRM
_RE_EFFUSIVE = re.compile(
    r"\bi'?m so proud|proud of you|inspiration to me|you got this|beautiful|beacon|"
    r"shining|warrior|hero of your|you are a (light|beacon)", re.I)


def _arms(arms):
    from . import discover_arms
    return discover_arms() if arms is None else arms


# ── 1. Oracle MITI behavior counts ───────────────────────────────────────────
def load_miti_behavior(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Per (arm, iteration, conversation) MITI behavior counts + R:Q ratio."""
    rows = []
    for arm in _arms(arms):
        for k in arm.iters:
            for fi, r in iter_conv_rows(arm.eval_dir(k, "MITI")):
                row = {"arm": arm.label, "method": arm.method, "K": arm.K,
                       "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                       "file_index": fi}
                for src, dst in _MITI_COLS.items():
                    row[dst] = float(r[src]) if src in r.index and pd.notna(r[src]) else None
                refl = (row.get("B4_SR") or 0) + (row.get("B5_CR") or 0)
                # R:Q is undefined when B3_Q is missing (not scored) OR a genuine 0 questions —
                # both correctly map to None (you can't form a reflection:question ratio with zero
                # questions); the falsy check covers both without a ZeroDivisionError.
                row["RtoQ"] = refl / row["B3_Q"] if row.get("B3_Q") else None
                # %CR per conversation (CR / all reflections), same None-on-empty convention.
                row["%CR"] = (row.get("B5_CR") or 0) / refl if refl else None
                # Official MITI 4.2.1 summary globals (manual §H): Technical = (CCT + SST)/2,
                # Relational = (Partnership + Empathy)/2 — the scores the manual's competency
                # thresholds are defined on (NOT our 4-global MITI_GlobalMean).
                cct, sst = row.get("ChangeTalk"), row.get("SoftenSustain")
                row["MITI_Technical"] = (cct + sst) / 2 if (cct is not None and sst is not None) else None
                par, emp = row.get("Partnership"), row.get("Empathy")
                row["MITI_Relational"] = (par + emp) / 2 if (par is not None and emp is not None) else None
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        df = _attach_by_arm(df, arms)
    return df


# Oracle MICI columns (negative-valence; higher = worse). Loaded only where scored.
_MICI_COLS = {
    "MICI_Severity": "MICI_Severity", "MICI_Rate": "MICI_Rate",
    "MICI_OverPraise": "MICI_OverPraise", "MICI_OverPraiseRate": "MICI_OverPraiseRate",
    "MICI_BehaviorTotal": "MICI_BehaviorTotal", "MICI_Confront": "MICI_Confront",
    "MICI_AdviseNoPermission": "MICI_AdviseNoPermission", "MICI_Warn": "MICI_Warn",
    "MICI_Direct": "MICI_Direct", "MICI_Judge": "MICI_Judge",
}


def load_mici_behavior(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Per (arm, iteration, conversation) MI-inconsistent behavior counts + rates.

    Empty (no rows) until the MICI questionnaire is scored via ``Run_Eval``.
    """
    rows = []
    for arm in _arms(arms):
        for k in arm.iters:
            for fi, r in iter_conv_rows(arm.eval_dir(k, "MICI")):
                row = {"arm": arm.label, "method": arm.method, "K": arm.K,
                       "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                       "file_index": fi}
                for src, dst in _MICI_COLS.items():
                    row[dst] = float(r[src]) if src in r.index and pd.notna(r[src]) else None
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        df = _attach_by_arm(df, arms)
    return df


# Oracle PCT columns (patient change-talk; 1-5 globals + patient-utterance counts). The 3 counts
# sum to PCT_BehaviorTotal (= patient-utterance count), so proportions are self-contained.
_PCT_COLS = {
    "PCT_Importance": "PCT_Importance", "PCT_Confidence": "PCT_Confidence",
    "PCT_Readiness": "PCT_Readiness", "PCT_GlobalMean": "PCT_GlobalMean",
    "PCT_ChangeTalk": "PCT_ChangeTalk", "PCT_SustainTalk": "PCT_SustainTalk",
    "PCT_Neutral": "PCT_Neutral", "PCT_BehaviorTotal": "PCT_BehaviorTotal",
    "PCT_ChangeProp": "PCT_ChangeProp",
}


def load_pct_behavior(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Per (arm, iteration, conversation) PCT globals + patient-utterance counts + ChangeProp.

    Empty (no rows) until the PCT questionnaire is scored via ``Run_Eval``.
    """
    rows = []
    for arm in _arms(arms):
        for k in arm.iters:
            for fi, r in iter_conv_rows(arm.eval_dir(k, "PCT")):
                row = {"arm": arm.label, "method": arm.method, "K": arm.K,
                       "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                       "file_index": fi}
                for src, dst in _PCT_COLS.items():
                    row[dst] = float(r[src]) if src in r.index and pd.notna(r[src]) else None
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        df = _attach_by_arm(df, arms)
    return df


def overpraise_crosscheck(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): the deterministic lexical over-praise marker rate beside the
    oracle's MICI_OverPraiseRate, so the regex's *direction* can be validated against the
    professional coder (same role ``loop%`` plays for degeneration).

    Returns an empty frame until MICI is scored. Columns: arm, method, K, iteration,
    lex_overpraise_marker_rate, MICI_OverPraiseRate.
    """
    text = text_metrics(arms, attach_persona=False)
    mici = load_mici_behavior(arms, attach_persona=False)
    if text.empty or mici.empty:
        return pd.DataFrame(columns=["arm", "method", "K", "iteration",
                                     "lex_overpraise_marker_rate", "MICI_OverPraiseRate"])
    keys = ["arm", "iteration", "file_index"]
    merged = text.merge(mici[keys + ["MICI_OverPraiseRate"]], on=keys, how="inner")
    return (merged.groupby(["arm", "method", "K", "iteration"], observed=True)
            [["lex_overpraise_marker_rate", "MICI_OverPraiseRate"]]
            .mean().reset_index().sort_values(["arm", "iteration"]))


def question_rate_crosscheck(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): the deterministic regex question rate ``q_per_turn`` (``?``-count per
    therapist turn) beside the oracle-derived ``q_per_turn_miti`` (MITI ``B3_Q`` / therapist turns).

    Unit-harmonized cross-check: both are questions-per-therapist-turn, one syntactic (literal
    ``?``), one the professional MITI question count — so they should track each other, and their
    late divergence (e.g. GRPO, when praise-heavy turns stop carrying a literal ``?``) is itself
    informative. Averages per-conversation ratios (mean-of-ratios). Returns an empty frame (no rows)
    until MITI is scored. Columns: arm, method, K, iteration, q_per_turn, q_per_turn_miti.
    """
    text = text_metrics(arms, attach_persona=False)
    miti = load_miti_behavior(arms, attach_persona=False)
    if text.empty or miti.empty:
        return pd.DataFrame(columns=["arm", "method", "K", "iteration",
                                     "q_per_turn", "q_per_turn_miti"])
    keys = ["arm", "iteration", "file_index"]
    merged = text.merge(miti[keys + ["B3_Q"]], on=keys, how="inner")
    # Sanity guard: the two rates MUST be built on the same per-conversation rows, else we would be
    # dividing an oracle count (B3_Q) by an unrelated conversation's turn count. The inner merge on
    # (arm, iteration, file_index) should keep ~every conversation that both sources scored; a large
    # drop means the MITI-eval index no longer aligns with the conversation index (e.g. a persona-
    # shuffle regression — see project-exp3-persona-shuffle-recovery). NOT a hard raise (thin/partial
    # arms legitimately shrink the join), just a visible warning.
    n_joinable = min(len(text[keys].drop_duplicates()), len(miti[keys].drop_duplicates()))
    if n_joinable and len(merged) < 0.9 * n_joinable:
        import warnings as _w
        _w.warn(f"question_rate_crosscheck: inner-join kept {len(merged)}/{n_joinable} conv rows "
                f"(<90%) — check MITI-eval vs conversation file_index alignment.", stacklevel=2)
    merged = merged[merged["n_th_turns"] > 0].copy()
    # Same denominator (n_th_turns) for both → the only difference is the numerator: literal '?' count
    # (q_per_turn) vs oracle question-function count (B3_Q). A widening gap (regex << MITI) is the
    # real affirmation/advice drift signature (declarative prompts carry no '?'), NOT a unit error.
    merged["q_per_turn_miti"] = merged["B3_Q"] / merged["n_th_turns"]
    return (merged.groupby(["arm", "method", "K", "iteration"], observed=True)
            [["q_per_turn", "q_per_turn_miti"]]
            .mean().reset_index().sort_values(["arm", "iteration"]))


# ── 2. Regex text metrics from conversations ─────────────────────────────────
def text_metrics(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Per (arm, iteration, conversation) text behavior metrics from the transcripts."""
    rows = []
    for arm in _arms(arms):
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
                th = cdf[cdf["role"] == "therapist"]["conversation"].astype(str).tolist()
                rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                             "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                             "file_index": int(m.group(1)), "conv_len": len(cdf),
                             **_turn_metrics(th)})
    df = pd.DataFrame(rows)
    if not df.empty and attach_persona:
        df = _attach_by_arm(df, arms)
    return df


def _turn_metrics(th: List[str]) -> dict:
    # Structural metrics (deterministic, primary) + two lexical-marker rates
    # (directional sanity-check on the oracle; renamed lex_* and kept out of the
    # headline _BEHAVIOR_METRICS — see module docstring).
    if not th:
        return {"n_th_turns": 0, "mean_turn_len": 0.0, "max_repeat": 0, "loop": False,
                "q_per_turn": 0.0, "lex_affirm_marker_rate": 0.0, "lex_overpraise_marker_rate": 0.0}
    counts = Counter(t.strip() for t in th)
    n = len(th)
    return {
        "n_th_turns": n,
        "mean_turn_len": sum(len(t) for t in th) / n,
        "max_repeat": max(counts.values()),
        "loop": max(counts.values()) >= 2,
        "q_per_turn": sum(t.count("?") for t in th) / n,
        "lex_affirm_marker_rate": sum(bool(_RE_AFFIRM.search(t)) for t in th) / n,
        "lex_overpraise_marker_rate": sum(bool(_RE_EFFUSIVE.search(t)) for t in th) / n,
    }


# ── Combined per-iteration trajectory ────────────────────────────────────────
# Headline behavior trajectory metrics. The semantic affirmation/over-praise signal is
# carried by the oracle-coded B6_AF (and MICI_OverPraiseRate once MICI is scored), NOT by the
# brittle lex_* marker rates, which stay out of this list (sanity-check only). All 7 MITI
# behaviors are covered (B1_GI/B7_Seek complete the set the drift grid used to omit).
_BEHAVIOR_METRICS = ["B3_Q", "B4_SR", "B5_CR", "B6_AF", "B2_Persuade", "B1_GI", "B7_Seek", "RtoQ",
                     "Empathy", "mean_turn_len", "loop", "q_per_turn", "conv_len"]

# Raw MITI behavior COUNTS that scale with conversation length. behavior_by_iter also emits a
# per-therapist-turn rate (`<m>_per_turn` = count / n_th_turns) for each, so trajectory figures
# aren't inflated by longer late-iteration conversations (the drift figure plots the rates).
_RATE_COUNT_METRICS = ["B3_Q", "B4_SR", "B5_CR", "B6_AF", "B2_Persuade", "B1_GI", "B7_Seek"]


def behavior_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration) means of every behavior metric — the trajectory backbone.

    Merges the oracle MITI counts and the regex text metrics (per conversation) then
    averages over conversations. ``loop`` becomes the fraction of degenerate convs.
    Parquet-cached (content-keyed on the eval + conversation CSVs; see :func:`~eda_analysis.data.load_cached`).
    """
    arms = _arms(arms)
    return load_cached("behavior_by_iter", arms, lambda: _behavior_by_iter_impl(arms),
                       input_roots=eval_input_roots(arms) + conv_input_roots(arms))


def _behavior_by_iter_impl(arms) -> pd.DataFrame:
    miti = load_miti_behavior(arms, attach_persona=False)
    text = text_metrics(arms, attach_persona=False)
    keys = ["arm", "method", "K", "model", "iteration", "is_base", "file_index"]
    if miti.empty and text.empty:
        return pd.DataFrame()
    if miti.empty:
        merged = text
    elif text.empty:
        merged = miti
    else:
        merged = text.merge(miti.drop(columns=["method", "K", "model", "is_base"]),
                            on=["arm", "iteration", "file_index"], how="outer")
    # Per-therapist-turn rates for the length-scaling MITI counts (mean-of-ratios, guarded on
    # n_th_turns>0 — same convention as question_rate_crosscheck). These are what the drift
    # figure plots so longer late-iteration convs don't mechanically inflate the counts.
    if "n_th_turns" in merged.columns:
        nt = merged["n_th_turns"].where(merged["n_th_turns"] > 0)
        for c in _RATE_COUNT_METRICS:
            if c in merged.columns:
                merged[f"{c}_per_turn"] = merged[c] / nt
    # Guard: an outer merge leaves method/K NaN for any conv scored by MITI but missing its
    # conversation CSV (e.g. partial Drive sync); the groupby below (dropna=True) would then drop
    # it silently. Warn rather than quietly shrink the per-iteration means.
    if "method" in merged.columns:
        n_orphan = int(merged["method"].isna().sum())
        if n_orphan:
            import warnings as _w
            _w.warn(f"behavior_by_iter: {n_orphan} MITI-scored conv row(s) have no matching "
                    f"conversation CSV (method/K NaN) — dropped from the per-iteration means; "
                    f"check conversation-file sync.", stacklevel=2)
    metrics = [m for m in _BEHAVIOR_METRICS if m in merged.columns]
    metrics += [f"{c}_per_turn" for c in _RATE_COUNT_METRICS if f"{c}_per_turn" in merged.columns]
    agg = (merged.groupby(["arm", "method", "K", "iteration"], observed=True)[metrics]
           .mean().reset_index().sort_values(["arm", "iteration"]))
    return agg


# ── MITI 4.2.1 proficiency summary scores (the official-threshold view) ───────────
# The four summary scores the MITI 4.2.1 manual defines competency thresholds on (constants
# .MITI_THRESHOLDS): R:Q, %CR, Technical global, Relational global. Per-conversation values come
# from load_miti_behavior (mean-of-ratios convention; ratio rows undefined on zero denominators
# are dropped from the mean, matching RtoQ handling elsewhere).
_PROFICIENCY_COLS = ["RtoQ", "%CR", "MITI_Technical", "MITI_Relational"]


def miti_proficiency_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): the 4 official MITI 4.2.1 summary scores (R:Q, %CR, Technical,
    Relational) — the frame the competency-threshold panel/table read.

    Columns: ``arm, method, K, iteration, is_base, R:Q, %CR, MITI_Technical, MITI_Relational``.
    Thresholds live in :data:`~eda_analysis.constants.MITI_THRESHOLDS` (fair, good) — expert
    opinion per the manual, and defined for ~20-min human sessions; caveat wherever drawn.
    Empty until MITI is scored. Parquet-cached.
    """
    arms = _arms(arms)
    return load_cached("miti_proficiency_by_iter", arms,
                       lambda: _miti_proficiency_by_iter_impl(arms),
                       input_roots=eval_input_roots(arms))


def _miti_proficiency_by_iter_impl(arms) -> pd.DataFrame:
    miti = load_miti_behavior(arms, attach_persona=False)
    if miti.empty:
        return pd.DataFrame()
    cols = [c for c in _PROFICIENCY_COLS if c in miti.columns]
    agg = (miti.groupby(["arm", "method", "K", "iteration", "is_base"], observed=True)[cols]
           .mean().reset_index().sort_values(["arm", "iteration"]))
    return agg.rename(columns={"RtoQ": "R:Q"})


# ── MICI per-item detail (severity global + per-therapist-turn behavior rates) ────
# The 6 MI-inconsistent behaviors are counts over therapist turns → per-turn rates (like the MITI
# drift grid). Higher = worse for every column. Severity is a 1-5 global; MICI_Rate is the total/turn.
_MICI_RATE_BEHAVIORS = ["MICI_Confront", "MICI_AdviseNoPermission", "MICI_Warn",
                        "MICI_Direct", "MICI_Judge", "MICI_OverPraise"]


def mici_behavior_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): MI-inconsistent severity + each harmful behavior PER THERAPIST TURN.

    The MICI analogue of :func:`behavior_by_iter`: joins the per-conversation MICI counts with the
    conversation therapist-turn count (from :func:`text_metrics`), forms a per-therapist-turn rate
    for each of the 6 behaviors (mean-of-ratios, guarded on ``n_th_turns > 0``), and averages over
    conversations. ``MICI_Severity`` (1-5 global) and ``MICI_Rate`` (total/turn) pass through as
    means. Empty until MICI is scored. Higher = worse for every column. Parquet-cached.
    """
    arms = _arms(arms)
    return load_cached("mici_behavior_by_iter", arms, lambda: _mici_behavior_by_iter_impl(arms),
                       input_roots=eval_input_roots(arms) + conv_input_roots(arms))


def _mici_behavior_by_iter_impl(arms) -> pd.DataFrame:
    mici = load_mici_behavior(arms, attach_persona=False)
    if mici.empty:
        return pd.DataFrame()
    text = text_metrics(arms, attach_persona=False)
    keys = ["arm", "iteration", "file_index"]
    rate_cols: List[str] = []
    if text.empty:
        merged = mici.copy()
    else:
        merged = mici.merge(text[keys + ["n_th_turns"]], on=keys, how="left")
        nt = merged["n_th_turns"].where(merged["n_th_turns"] > 0)
        for b in _MICI_RATE_BEHAVIORS:
            if b in merged.columns:
                merged[f"{b}_rate"] = merged[b] / nt
                rate_cols.append(f"{b}_rate")
    val_cols = [c for c in ["MICI_Severity", "MICI_Rate"] if c in merged.columns] + rate_cols
    return (merged.groupby(["arm", "method", "K", "iteration"], observed=True)[val_cols]
            .mean().reset_index().sort_values(["arm", "iteration"]))


# ── PCT per-item detail (patient globals + utterance-type proportions) ────────────
# The 3 counts (ChangeTalk/SustainTalk/Neutral) sum to PCT_BehaviorTotal (= patient utterances),
# so each becomes a proportion of patient utterances — the patient-side analogue of the per-turn
# rate. Higher = better for every column EXCEPT PCT_SustainTalk_prop.
_PCT_PROP_BEHAVIORS = ["PCT_ChangeTalk", "PCT_SustainTalk", "PCT_Neutral"]


def pct_behavior_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration): patient change-talk globals (1-5) + utterance-type PROPORTIONS.

    The 3 patient globals (Importance/Confidence/Readiness) + the derived ChangeProp pass through
    as means; each of the 3 utterance counts is turned into a proportion of patient utterances
    (``count / PCT_BehaviorTotal``, mean-of-ratios). Empty until PCT is scored. Parquet-cached.
    """
    arms = _arms(arms)
    return load_cached("pct_behavior_by_iter", arms, lambda: _pct_behavior_by_iter_impl(arms),
                       input_roots=eval_input_roots(arms) + conv_input_roots(arms))


def _pct_behavior_by_iter_impl(arms) -> pd.DataFrame:
    pct = load_pct_behavior(arms, attach_persona=False)
    if pct.empty:
        return pd.DataFrame()
    prop_cols: List[str] = []
    if "PCT_BehaviorTotal" in pct.columns:
        tot = pct["PCT_BehaviorTotal"].where(pct["PCT_BehaviorTotal"] > 0)
        for b in _PCT_PROP_BEHAVIORS:
            if b in pct.columns:
                pct[f"{b}_prop"] = pct[b] / tot
                prop_cols.append(f"{b}_prop")
    val_cols = [c for c in ["PCT_Importance", "PCT_Confidence", "PCT_Readiness",
                            "PCT_GlobalMean", "PCT_ChangeProp"] if c in pct.columns] + prop_cols
    return (pct.groupby(["arm", "method", "K", "iteration"], observed=True)[val_cols]
            .mean().reset_index().sort_values(["arm", "iteration"]))


def _attach_by_arm(df: pd.DataFrame, arms) -> pd.DataFrame:
    seed_by_arm = {a.label: a.seed for a in _arms(arms)}
    parts = [attach_personas(g, seed_by_arm.get(lab, 42)) for lab, g in df.groupby("arm", sort=False)]
    return pd.concat(parts, ignore_index=True)
