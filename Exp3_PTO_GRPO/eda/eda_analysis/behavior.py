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
affirmation/over-praise story; see notebook ``2_Behavior_and_Mechanism`` for the cross-check.
"""

import os
import re
from collections import Counter
from typing import List, Optional

import pandas as pd

from .data import attach_personas  # personas merged into data.py

_MITI_COLS = {
    "MITI_B1_GI": "B1_GI", "MITI_B3_Q": "B3_Q", "MITI_B4_SR": "B4_SR", "MITI_B5_CR": "B5_CR",
    "MITI_B6_AF": "B6_AF", "MITI_B7_Seek": "B7_Seek", "MITI_B2_Persuade": "B2_Persuade",
    "MITI4_Empathy": "Empathy", "MITI1_CultivatingChangeTalk": "ChangeTalk",
    "MITI3_Partnership": "Partnership", "MITI_GlobalMean": "MITI_Global",
}

# Lexical marker cues (case-insensitive), matched per therapist turn. These are a
# DIRECTIONAL sanity-check on the oracle's affirmation / over-praise counts, NOT primary
# metrics — see the module docstring.
_RE_AFFIRM = re.compile(r"\byou are\b|\byou're (worthy|enough|strong|powerful|brave|amazing|a )", re.I)
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
            ddir = arm.eval_dir(k, "MITI")
            if not os.path.isdir(ddir):
                continue
            for fn in os.listdir(ddir):
                stem, ext = os.path.splitext(fn)
                if ext != ".csv" or not stem.isdigit():
                    continue
                try:
                    r = pd.read_csv(os.path.join(ddir, fn)).iloc[0]
                except Exception:
                    continue
                row = {"arm": arm.label, "method": arm.method, "K": arm.K,
                       "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                       "file_index": int(stem)}
                for src, dst in _MITI_COLS.items():
                    row[dst] = float(r[src]) if src in r.index and pd.notna(r[src]) else None
                refl = (row.get("B4_SR") or 0) + (row.get("B5_CR") or 0)
                row["RtoQ"] = refl / row["B3_Q"] if row.get("B3_Q") else None
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
            ddir = arm.eval_dir(k, "MICI")
            if not os.path.isdir(ddir):
                continue
            for fn in os.listdir(ddir):
                stem, ext = os.path.splitext(fn)
                if ext != ".csv" or not stem.isdigit():
                    continue
                try:
                    r = pd.read_csv(os.path.join(ddir, fn)).iloc[0]
                except Exception:
                    continue
                row = {"arm": arm.label, "method": arm.method, "K": arm.K,
                       "model": arm.model_name(k), "iteration": k, "is_base": (k == 0),
                       "file_index": int(stem)}
                for src, dst in _MICI_COLS.items():
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
# brittle lex_* marker rates, which stay out of this list (sanity-check only).
_BEHAVIOR_METRICS = ["B3_Q", "B4_SR", "B5_CR", "B6_AF", "B2_Persuade", "RtoQ",
                     "Empathy", "mean_turn_len", "loop", "q_per_turn", "conv_len"]


def behavior_by_iter(arms: Optional[List] = None) -> pd.DataFrame:
    """Per (arm, iteration) means of every behavior metric — the trajectory backbone.

    Merges the oracle MITI counts and the regex text metrics (per conversation) then
    averages over conversations. ``loop`` becomes the fraction of degenerate convs.
    """
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
    metrics = [m for m in _BEHAVIOR_METRICS if m in merged.columns]
    agg = (merged.groupby(["arm", "method", "K", "iteration"], observed=True)[metrics]
           .mean().reset_index().sort_values(["arm", "iteration"]))
    return agg


def _attach_by_arm(df: pd.DataFrame, arms) -> pd.DataFrame:
    seed_by_arm = {a.label: a.seed for a in _arms(arms)}
    parts = [attach_personas(g, seed_by_arm.get(lab, 42)) for lab, g in df.groupby("arm", sort=False)]
    return pd.concat(parts, ignore_index=True)
