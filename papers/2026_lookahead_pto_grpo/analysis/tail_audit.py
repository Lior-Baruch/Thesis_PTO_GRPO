"""tail_audit.py — what the K-step look-ahead reward is actually scoring.

An audit of the look-ahead TAILS (the K=5 simulated turns appended to each candidate before the
oracle scores it; only the LA5 arms carry them) plus the API-call cost axis for all four arms.

Data: ``iteration_N/eda/generations.jsonl`` per arm (one record per branch/group, candidates nested,
each with ``lookahead{k, realized_turns, ended_early, tail}``), read through
``eda_analysis.training.load_generations(keep_tail=True)`` one arm at a time; a light second
streaming pass collects the record-level fields that loader drops (prefix chars, oracle retries);
the ``model_iter_*`` conversation CSVs give the eval-conv patient turns.

Facts established while inspecting the rows (see the scout notes in the paper's NUMBERS.md):
* ``tail`` is the ORACLE-FORMAT transcript slice appended after the completion —
  ``"\\n\\n[PATIENT]: ...\\n\\n[THERAPIST]: ..."`` — starting with the patient's reply.
* ``realized_turns`` == number of ``[ROLE]:`` labels in the tail (0..5); ``ended_early`` ==
  ``realized_turns < 5``. Odd counts end on a PATIENT turn, even (>0) on a THERAPIST turn.
* The literal ``"SESSION ENDED"`` marker never appears in a tail: ``reward.simulate_lookahead_batch``
  runs ``convs.handle_session_end`` which STRIPS the marker and keeps only the text before it.
  What survives is a fingerprint — the kept text ends in whitespace (the marker followed) — which
  in the eval CSVs separates patient-closed turns from ordinary ones at ~100% vs ~4%.
* PTO candidates whose completion is EMPTY carry ``lookahead=None`` and ``score=None`` (never
  simulated, never scored) — they are dropped from the tail analysis and cost no calls.
* GRPO records also carry a ``phase == "eval"`` block (TRL's eval loop) — those candidates were
  scored (real calls) but produced no gradient; they are dropped from the training-signal
  analysis and kept in the API accounting.

Outputs (all prefixed ``tail_audit_``): tables ``by_iter``, ``cues_by_iter``, ``within_group``,
``api_calls``, ``api_ratio``; figures ``fig`` (3 panels) and ``fig_api`` (2 panels); ledger
``out/tail_audit.json``.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

from eda_analysis import EdaConfig  # noqa: E402
from eda_analysis.config import cross_k_arms  # noqa: E402
from eda_analysis import training as T  # noqa: E402
from eda_analysis.constants import RE_AFFIRM, RE_EFFUSIVE  # noqa: E402

SCRIPT = "tail_audit"
K_LA = 5
N_Q = 2                     # Q1 + Q2 → two oracle calls per scored candidate
N_EVAL_RUBRICS = 8          # Run_Eval scores every model state on 8 instruments (per grader)
N_BOOT = 1000
SEED = 0
GROUP_KEYS = ["train_iter", "conversation_id", "branch_id", "epoch"]   # + arm (one arm at a time)
_ROLE_RE = re.compile(r"\[(THERAPIST|PATIENT)\]: ")
_LEAK = "<|im_start|>"
# wrap-up cues in a PATIENT turn — used only to VALIDATE the whitespace fingerprint of a stripped SESSION ENDED
_WRAPUP_RE = re.compile(r"wrap|for now|next time|next session|enough for today|i think i'?m good|call it|"
                        r"end (the|this) session|good place to stop|thank", re.I)

L = C.Ledger(SCRIPT)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_tail(tail: str):
    """Split an oracle-format tail into [(role, text), ...] in order."""
    if not tail:
        return []
    ms = list(_ROLE_RE.finditer(tail))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(tail)
        out.append((m.group(1), tail[m.end():end]))
    return out


def end_reason(rt, tail: str) -> str:
    """Classify how a K=5 look-ahead tail terminated (see module docstring for the fingerprint)."""
    if rt is None:
        return "not_simulated"
    rt = int(rt)
    if rt >= K_LA:
        return "full"
    if rt == 0:
        return "no_tail"                      # patient call failed / bare marker as first reply
    ws = bool(tail) and (tail != tail.rstrip())
    if rt % 2 == 1:                           # ends on a PATIENT turn
        return "patient_closed" if ws else "therapist_stalled"
    return "after_therapist"                  # ends on a THERAPIST turn (therapist marker / patient fail)


def tail_features(row) -> dict:
    """Per-candidate tail structure + cue features (therapist turns in the tail vs the candidate)."""
    tail = row["tail"] if isinstance(row["tail"], str) else ""
    turns = parse_tail(tail)
    pat = [t for r, t in turns if r == "PATIENT"]
    th = [t for r, t in turns if r == "THERAPIST"]
    comp = row["completion"] if isinstance(row["completion"], str) else ""
    th_strip = [t.strip() for t in th if t.strip()]
    pool = [comp.strip()] + th_strip if comp.strip() else th_strip
    cnt = Counter(pool)
    max_rep = max(cnt.values()) if cnt else 0
    n_th = len(th)
    return {
        "tail_chars": len(tail),
        "tail_patient_turns": len(pat),
        "tail_therapist_turns": n_th,
        "tail_patient_chars": sum(len(t) for t in pat),
        "tail_th_chars": sum(len(t) for t in th),
        "tail_th_turn_len": (sum(len(t) for t in th) / n_th) if n_th else np.nan,
        "tail_th_q_per_turn": (sum(t.count("?") for t in th) / n_th) if n_th else np.nan,
        "tail_th_affirm_rate": (sum(bool(RE_AFFIRM.search(t)) for t in th) / n_th) if n_th else np.nan,
        "tail_th_effusive_rate": (sum(bool(RE_EFFUSIVE.search(t)) for t in th) / n_th) if n_th else np.nan,
        "tail_loop": bool(max_rep >= 2),               # a tail therapist turn repeats the candidate / another tail turn
        "tail_leak": _LEAK in tail,
        "tail_ws_end": bool(tail) and (tail != tail.rstrip()),
        "last_patient_wrapup_cue": bool(pat) and bool(_WRAPUP_RE.search(pat[-1])) and (turns[-1][0] == "PATIENT"),
        "cand_q": comp.count("?"),
        "cand_affirm": bool(RE_AFFIRM.search(comp)),
        "cand_effusive": bool(RE_EFFUSIVE.search(comp)),
        "end_reason": end_reason(row["realized_turns"], tail),
    }


def boot_ci(values, stat=np.mean, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    b = stat(v[idx], axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def stream_record_stats(arm) -> pd.DataFrame:
    """Second (light) pass over an arm's generations.jsonl: record-level fields load_generations
    drops — prefix chars + oracle retries + per-candidate char sizes — aggregated per
    (train_iter, phase). No text is retained."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(arm.runs_dir, "iteration_*", "eda", "generations.jsonl"))):
        acc = {}
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (rec.get("iteration"), rec.get("phase"))
                a = acc.setdefault(key, Counter())
                pre = len(rec.get("prefix") or "")
                a["n_records"] += 1
                for c in rec.get("candidates", []) or []:
                    comp = c.get("completion") or ""
                    la = c.get("lookahead") or None
                    orc = c.get("oracle") or {}
                    a["n_cands"] += 1
                    if la is None:                        # PTO empty completion: not simulated, not scored
                        a["n_not_scored"] += 1
                        continue
                    sub = c.get("sub_scores") or {}
                    n_calls = len(sub) if c.get("score") is None else N_Q
                    if c.get("score") is None and not sub:
                        n_calls = 1                       # first oracle call failed outright
                    a["n_scored"] += int(c.get("score") is not None)
                    a["oracle_calls"] += n_calls + int(orc.get("retries") or 0)
                    a["oracle_retries"] += int(orc.get("retries") or 0)
                    tail = la.get("tail") or ""
                    doc = pre + len("\n\n[THERAPIST]: ") + len(comp) + len(tail)
                    a["oracle_input_chars"] += doc * n_calls
                    rt = int(la.get("realized_turns") or 0)
                    n_pat = int(np.ceil(rt / 2)) if rt > 0 else (1 if (la.get("k") or 0) > 0 else 0)
                    a["patient_calls_tail"] += n_pat
                    a["patient_input_chars_tail_ub"] += n_pat * doc   # upper bound: each call sees ≤ the full doc
                    a["therapist_gens_tail"] += rt // 2
        for (it, ph), a in acc.items():
            rows.append({"arm": arm.label, "method": arm.method, "K": arm.K, "train_iter": it, "phase": ph, **a})
    return pd.DataFrame(rows)


GRPO_GROUPS_PER_STEP = 16       # generation_batch 128 completions per optimizer step / G=8 siblings
GRPO_CANDS_PER_STEP = 128


def grpo_steps(arm) -> dict:
    """GRPO ground truth for how many optimizer steps each iteration ran: one
    ``training/completions/*.parquet`` per step (== the last checkpoint number). The EDA capture in
    ``generations.jsonl`` is per-PROCESS on the older iterations — a crashed+resumed iteration logged only
    its post-resume steps (GRPO_LA5 iters 1-2, GRPO_LA0 iters 2/6/8) — so calls are scaled by
    ``logged_groups / (steps x 16)``. Empty for PTO (whose Step-2 checkpoint carries the records)."""
    out = {}
    if arm.method != "GRPO":
        return out
    for d in glob.glob(os.path.join(arm.runs_dir, "iteration_*")):
        m = re.search(r"iteration_(\d+)$", d.replace("\\", "/"))
        if not m:
            continue
        n = len(glob.glob(os.path.join(d, "training", "completions", "*.parquet")))
        if n and os.path.exists(os.path.join(d, "eda", "generations.jsonl")):
            out[int(m.group(1))] = n
    return out


def eval_conv_stats(arm) -> pd.DataFrame:
    """Per model_iter: patient/therapist turn + char counts and session-end reasons from the CSVs."""
    rows = []
    for k in arm.iters:
        cdir = arm.conv_dir(k)
        if not cdir or not os.path.isdir(cdir):
            continue
        n_conv = 0; pat = 0; th = 0; pchars = 0; tchars = 0
        ended = Counter(); conv_len = []
        for fn in os.listdir(cdir):
            if not re.match(r"conversation_\d+\.csv$", fn):
                continue
            try:
                cdf = pd.read_csv(os.path.join(cdir, fn))
            except Exception:
                continue
            n_conv += 1
            r = cdf["role"].astype(str)
            txt = cdf["conversation"].fillna("").astype(str)
            pat += int((r == "patient").sum()); th += int((r == "therapist").sum())
            pchars += int(txt[r == "patient"].str.len().sum()); tchars += int(txt[r == "therapist"].str.len().sum())
            conv_len.append(len(cdf))
            eb = cdf["session_ended_by"].dropna() if "session_ended_by" in cdf.columns else pd.Series(dtype=object)
            ended[str(eb.iloc[0]) if len(eb) else "none"] += 1
        rows.append({"arm": arm.label, "method": arm.method, "K": arm.K, "model_iter": k, "n_convs": n_conv,
                     "eval_patient_turns": pat, "eval_therapist_turns": th,
                     "eval_patient_chars": pchars, "eval_therapist_chars": tchars,
                     "eval_conv_len_mean": float(np.mean(conv_len)) if conv_len else np.nan,
                     "eval_ended_by_patient": ended.get("patient", 0),
                     "eval_ended_by_therapist": ended.get("therapist", 0),
                     "eval_ended_none": ended.get("none", 0)})
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    C.style()
    pal = C.palette()
    arms = cross_k_arms(EdaConfig(view="L5", verbose=False))
    by_label = {a.label: a for a in arms}
    la5 = [by_label[l] for l in ("PTO_LA5", "GRPO_LA5") if l in by_label]
    print("arms:", [a.label for a in arms])

    # ════════════════════════════════════════════════════════════════════════
    # Part 1 + 2 — tail audit on the LA5 arms (one arm at a time; memo cleared after each)
    # ════════════════════════════════════════════════════════════════════════
    by_iter_rows, cues_rows, wg_rows, rt_curve_rows, chosen_rows = [], [], [], [], []
    scout_check = {}
    for arm in la5:
        print(f"[{arm.label}] load_generations(keep_tail=True) …")
        g = T.load_generations([arm], keep_tail=True)
        T.clear_generations_memo()
        n_all = len(g)
        n_eval_phase = int((g["phase"] == "eval").sum())
        g = g[(g["phase"] != "eval")].copy()
        n_train_phase = len(g)
        n_not_sim = int(g["realized_turns"].isna().sum())         # PTO empty completions (lookahead=None)
        pre_counts = (g.assign(unscored=g["realized_turns"].notna() & g["score"].isna(),
                               not_sim=g["realized_turns"].isna())
                      .groupby("train_iter")[["unscored", "not_sim"]].sum())
        steps = grpo_steps(arm)
        g = g[g["realized_turns"].notna() & g["score"].notna()].copy()
        g["realized_turns"] = g["realized_turns"].astype(int)
        g["ended_early"] = g["ended_early"].astype(bool)
        g["epoch"] = g["epoch"].fillna(-1.0)
        print(f"   rows: all={n_all} eval_phase={n_eval_phase} train_phase={n_train_phase} "
              f"not_simulated={n_not_sim} analysed={len(g)}")
        L.put(f"rows.{arm.label}", {"all": n_all, "eval_phase_dropped": n_eval_phase,
                                    "not_simulated_dropped": n_not_sim, "unscored_dropped": n_train_phase - n_not_sim - len(g),
                                    "analysed": len(g)},
              source="tail_audit.py stdout; generations.jsonl row filter")

        # scout cross-check: GRPO_LA5 iteration 5, first 300 groups (2,400 candidates)
        if arm.label == "GRPO_LA5":
            g5 = g[g["train_iter"] == 5]
            first_groups = g5.drop_duplicates(GROUP_KEYS)[GROUP_KEYS].head(300)
            sub = g5.merge(first_groups, on=GROUP_KEYS)
            scout_check = {"n": int(len(sub)),
                           "realized_turns": {int(k): int(v) for k, v in sub["realized_turns"].value_counts().items()},
                           "ended_early": int(sub["ended_early"].sum()),
                           "ended_early_rate": float(sub["ended_early"].mean())}
            print("   scout cross-check (GRPO_LA5 iter 5, first 300 groups):", scout_check)

        feats = pd.DataFrame([tail_features(r) for r in g[["tail", "completion", "realized_turns"]].to_dict("records")],
                             index=g.index)
        g = pd.concat([g.drop(columns=["tail"]), feats], axis=1)

        # within-group quantities: deviation from group mean, z (= GRPO's advantage), group size
        grp = g.groupby(GROUP_KEYS, observed=True)["score"]
        g["group_n"] = grp.transform("size")
        g["dev"] = g["score"] - grp.transform("mean")
        sd = grp.transform("std").replace(0.0, np.nan)
        g["z"] = g["dev"] / sd
        # a chosen candidate = argmax score (GRPO chosen_idx; PTO role=='chosen' — identical, checked)
        g["chosen"] = g["is_chosen"].astype(bool)

        iters = sorted(g["train_iter"].unique())
        for it in iters + ["pooled"]:
            d = g if it == "pooled" else g[g["train_iter"] == it]
            n = len(d)
            n_groups = d.drop_duplicates(GROUP_KEYS).shape[0]
            rt_counts = d["realized_turns"].value_counts()
            ee = d["ended_early"].astype(float)
            ee_lo, ee_hi = boot_ci(ee.values)
            er = d["end_reason"].value_counts(normalize=True)
            if it == "pooled":
                cov = (sum(d.drop_duplicates(GROUP_KEYS).groupby("train_iter").size().get(i, 0) for i in steps)
                       / (GRPO_GROUPS_PER_STEP * sum(steps.values()))) if steps else 1.0
                n_unsc, n_nsim = int(pre_counts["unscored"].sum()), int(pre_counts["not_sim"].sum())
            else:
                cov = (n_groups / (GRPO_GROUPS_PER_STEP * steps[it])) if (steps and it in steps) else 1.0
                n_unsc = int(pre_counts["unscored"].get(it, 0)); n_nsim = int(pre_counts["not_sim"].get(it, 0))
            row = {"arm": arm.label, "train_iter": it, "n_groups": n_groups, "n_candidates": n,
                   "log_coverage": cov, "n_unscored_dropped": n_unsc, "n_not_simulated_dropped": n_nsim,
                   "mean_score": d["score"].mean(),
                   "realized_turns_mean": d["realized_turns"].mean(),
                   **{f"rt{k}_share": float(rt_counts.get(k, 0)) / n for k in range(6)},
                   "ended_early_rate": ee.mean(), "ended_early_ci_lo": ee_lo, "ended_early_ci_hi": ee_hi,
                   "patient_closed_share": float(er.get("patient_closed", 0.0)),
                   "therapist_stalled_share": float(er.get("therapist_stalled", 0.0)),
                   "after_therapist_share": float(er.get("after_therapist", 0.0)),
                   "no_tail_share": float(er.get("no_tail", 0.0)),
                   "full_share": float(er.get("full", 0.0)),
                   "full_closed_at_turn5_share": float((d["tail_ws_end"] & (d["realized_turns"] == K_LA)).mean()),
                   "wrapup_cue_rate_patient_closed": float(d.loc[d["end_reason"] == "patient_closed", "last_patient_wrapup_cue"].mean()) if (d["end_reason"] == "patient_closed").any() else np.nan,
                   "wrapup_cue_rate_full_open": float(d.loc[(d["realized_turns"] == K_LA) & ~d["tail_ws_end"], "last_patient_wrapup_cue"].mean()),
                   "tail_chars_mean": d["tail_chars"].mean(),
                   "tail_patient_turns_mean": d["tail_patient_turns"].mean(),
                   "tail_therapist_turns_mean": d["tail_therapist_turns"].mean(),
                   "tail_loop_rate": d["tail_loop"].astype(float).mean(),
                   "tail_leak_rate": d["tail_leak"].astype(float).mean(),
                   "cand_floored_rate": d["floored"].astype(float).mean(),
                   "cand_empty_rate": d["empty"].astype(float).mean(),
                   "cand_leak_rate": d["leak"].astype(float).mean()}
            by_iter_rows.append(row)
            cues_rows.append({"arm": arm.label, "train_iter": it, "n_candidates": n,
                              "n_tail_th_turns": int(d["tail_therapist_turns"].sum()),
                              "cand_len_chars": d["len_chars"].mean(),
                              "tail_th_turn_len_chars": d["tail_th_turn_len"].mean(),
                              "cand_q_per_turn": d["cand_q"].mean(),
                              "tail_th_q_per_turn": d["tail_th_q_per_turn"].mean(),
                              "cand_affirm_rate": d["cand_affirm"].astype(float).mean(),
                              "tail_th_affirm_rate": d["tail_th_affirm_rate"].mean(),
                              "cand_effusive_rate": d["cand_effusive"].astype(float).mean(),
                              "tail_th_effusive_rate": d["tail_th_effusive_rate"].mean(),
                              "tail_loop_rate": d["tail_loop"].astype(float).mean(),
                              "tail_patient_turn_len_chars": (d["tail_patient_chars"].sum() / max(1, d["tail_patient_turns"].sum()))})

            # ── within-group: does the tail predict the reward? ──
            gk = d.groupby(GROUP_KEYS, observed=True)
            # (i) pooled group-demeaned Spearman: score dev vs realized_turns / tail chars (groups with variance)
            rt_dev = d["realized_turns"] - gk["realized_turns"].transform("mean")
            has_var = gk["realized_turns"].transform("nunique") > 1
            rho_rt = sps.spearmanr(d.loc[has_var, "dev"], rt_dev[has_var]).statistic if has_var.sum() > 10 else np.nan
            tc_dev = d["tail_chars"] - gk["tail_chars"].transform("mean")
            rho_tc = sps.spearmanr(d["dev"], tc_dev).statistic if len(d) > 10 else np.nan
            # (ii) paired: mean(score | ended_early) − mean(score | full) within groups holding both
            dd = d.assign(s_ee=d["score"].where(d["ended_early"]), s_full=d["score"].where(~d["ended_early"]),
                          z_ee=d["z"].where(d["ended_early"]), z_full=d["z"].where(~d["ended_early"]),
                          n_ee=d["ended_early"].astype(int), n_full=(~d["ended_early"]).astype(int),
                          ch_ee=(d["chosen"] & d["ended_early"]).astype(int),
                          ch_full=(d["chosen"] & ~d["ended_early"]).astype(int))
            per = dd.groupby(GROUP_KEYS, observed=True).agg(
                m_ee=("s_ee", "mean"), m_full=("s_full", "mean"), z_ee=("z_ee", "mean"), z_full=("z_full", "mean"),
                n_ee=("n_ee", "sum"), n_full=("n_full", "sum"), ch_ee=("ch_ee", "sum"), ch_full=("ch_full", "sum")).reset_index()
            both = per[(per["n_ee"] > 0) & (per["n_full"] > 0)]
            pr = C.paired(both["m_ee"].values, both["m_full"].values, n_boot=N_BOOT, seed=SEED)
            # (iii) P(chosen | ended_early) vs P(chosen | full) per candidate; P(argmax ended early) vs base rate
            #       per group — both with a bootstrap over GROUPS (the sampling unit)
            A = per[["n_ee", "n_full", "ch_ee", "ch_full"]].to_numpy(float)
            def _stats(M):
                n_ee_, n_full_, ch_ee_, ch_full_ = M[..., 0], M[..., 1], M[..., 2], M[..., 3]
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_ee = ch_ee_ / n_ee_; p_full = ch_full_ / n_full_
                    rr_ = p_ee / p_full
                    diff_ = ch_ee_ / (ch_ee_ + ch_full_) - n_ee_ / (n_ee_ + n_full_)
                return p_ee, p_full, rr_, diff_
            p_ch_ee, p_ch_full, rr, diff = _stats(A.sum(axis=0))
            rng = np.random.default_rng(SEED)
            ng = A.shape[0]
            rr_b, diff_b = [], []
            for _ in range(N_BOOT // 100):            # chunked group bootstrap (memory)
                idx = rng.integers(0, ng, size=(100, ng))
                _, _, r_b, d_b = _stats(A[idx].sum(axis=1))
                rr_b.append(r_b); diff_b.append(d_b)
            rr_b = np.concatenate(rr_b); diff_b = np.concatenate(diff_b)
            p_ch_ee, p_ch_full, rr = float(p_ch_ee), float(p_ch_full), float(rr)
            rr_lo, rr_hi = np.nanpercentile(rr_b, [2.5, 97.5])
            df_lo, df_hi = np.nanpercentile(diff_b, [2.5, 97.5])
            p_chosen_is_ee = float(per["ch_ee"].sum() / max(1, (per["ch_ee"] + per["ch_full"]).sum()))
            base_ee = float(d["ended_early"].mean())
            n_groups_rt_var = int(d.loc[has_var].drop_duplicates(GROUP_KEYS).shape[0]) if has_var.any() else 0
            wg_rows.append({"arm": arm.label, "train_iter": it, "n_groups": n_groups, "n_candidates": n, "log_coverage": cov,
                            "rho_dev_vs_realized_turns": rho_rt, "n_groups_rt_var": n_groups_rt_var,
                            "rho_dev_vs_tail_chars": rho_tc,
                            "n_groups_both": pr["n"], "delta_ee_minus_full": pr["mean_delta"], "dz": pr["dz"],
                            "ci_lo": pr["ci_lo"], "ci_hi": pr["ci_hi"], "p": pr["p"],
                            "z_ee_mean": both["z_ee"].mean() if len(both) else np.nan,
                            "z_full_mean": both["z_full"].mean() if len(both) else np.nan,
                            "base_ee_rate": base_ee, "p_chosen_is_ee": p_chosen_is_ee,
                            "chosen_minus_base": p_chosen_is_ee - base_ee, "chosen_minus_base_ci_lo": df_lo, "chosen_minus_base_ci_hi": df_hi,
                            "p_chosen_given_ee": p_ch_ee, "p_chosen_given_full": p_ch_full,
                            "rr_chosen_ee_vs_full": rr, "rr_ci_lo": rr_lo, "rr_ci_hi": rr_hi})
            chosen_rows.append({"arm": arm.label, "train_iter": it, "p_chosen_given_ee": p_ch_ee,
                                "p_chosen_given_full": p_ch_full, "rr": rr, "rr_lo": rr_lo, "rr_hi": rr_hi})
            # curve for fig (b): mean within-group deviation by realized_turns (pooled + per iter)
            for k, s in d.groupby("realized_turns")["dev"]:
                if len(s) >= 5:
                    lo, hi = boot_ci(s.values, n_boot=500)
                    rt_curve_rows.append({"arm": arm.label, "train_iter": it, "realized_turns": int(k), "n": len(s),
                                          "dev_mean": s.mean(), "dev_lo": lo, "dev_hi": hi,
                                          "score_mean": d.loc[s.index, "score"].mean(),
                                          "p_chosen": d.loc[s.index, "chosen"].mean()})
        del g, feats

    by_iter = pd.DataFrame(by_iter_rows)
    cues = pd.DataFrame(cues_rows)
    wg = pd.DataFrame(wg_rows)
    rt_curve = pd.DataFrame(rt_curve_rows)
    # Holm within (arm) family across iterations (pooled row excluded from the family)
    wg["p_holm"] = np.nan
    for arm_l, idx in wg[wg["train_iter"] != "pooled"].groupby("arm").groups.items():
        wg.loc[idx, "p_holm"] = C.holm(wg.loc[idx, "p"].values)

    C.save_table(by_iter, f"{SCRIPT}_by_iter", nd=3, caption=(
        "Look-ahead TAIL structure per LA5 arm x training iteration (train_iter n = the branching done by policy "
        "pi_n, whose eval convs are model_iter_{n-1}; there is no tail at iteration 0 because look-ahead only runs "
        "during training). One row per candidate scored under K=5 (GRPO 'eval'-phase groups and PTO empty "
        "completions excluded); realized_turns = number of simulated turns in the tail (0..5), ended_early = "
        "realized_turns < 5. rtK_share = share of candidates with K realized turns; the end-reason shares "
        "classify how the tail stopped: patient_closed = the patient wrote SESSION ENDED (the marker is stripped "
        "by handle_session_end; identified by the trailing-whitespace fingerprint), therapist_stalled = an "
        "empty/degenerate therapist turn froze the sim, after_therapist = ended on a therapist turn, no_tail = "
        "zero simulated turns; full_closed_at_turn5 = the fingerprint on a full 5-turn tail. wrapup_cue_rate_* "
        "validates the fingerprint: share of tails whose LAST patient turn carries a wrap-up phrase (wrap / for now / "
        "next time / thank ...) among patient_closed tails vs among full tails that did not end in whitespace. tail_loop = a "
        "therapist turn in the tail verbatim-repeats the candidate or another tail turn. Bootstrap 95% CI over "
        "candidates. log_coverage = logged groups / (optimizer steps x 16) for GRPO — iterations that crashed and "
        "resumed logged only their post-resume steps (GRPO_LA5 iters 1-2 ~0.5/0.7), so those rows describe the "
        "later half of the iteration; PTO logs are complete (1.0). n_unscored_dropped = candidates with a tail but "
        "no oracle score (an oracle-API incident at PTO_LA5 iteration 5); n_not_simulated_dropped = PTO empty "
        "completions. GRPO_LA5 is right-censored at iteration 5. Grader: the training oracle (gpt-4o-mini)."))
    C.save_table(cues, f"{SCRIPT}_cues_by_iter", nd=3, caption=(
        "Simple lexical cues in the look-ahead tail's THERAPIST turns versus in the scored candidate itself, per "
        "LA5 arm x train_iter (rates per therapist turn; question marks per turn, the RE_AFFIRM / RE_EFFUSIVE "
        "regexes from eda_analysis.constants — directional sanity cues, not primary metrics). "
        "GRPO_LA5 ends at iteration 5. Grader: training oracle (gpt-4o-mini)."))
    C.save_table(wg, f"{SCRIPT}_within_group", nd=3, caption=(
        "Does the tail predict the reward WITHIN a group (the unit the update sees: GRPO's G=8 siblings, PTO's M=8 "
        "branches)? rho_dev_vs_* = Spearman of the group-demeaned score against the group-demeaned realized_turns "
        "/ tail chars (pooled over groups with within-group variance). delta_ee_minus_full = mean(score | ended "
        "early) - mean(score | full 5-turn tail) paired within groups holding both kinds (SIGN: + => the "
        "ended-early candidates score HIGHER); dz, bootstrap 95% CI, Wilcoxon p, Holm within arm across "
        "iterations. z_* = the group-standardised score (GRPO's advantage). p_chosen_is_ee = share of groups "
        "whose argmax candidate ended early vs base_ee_rate = share of all candidates that ended early; "
        "p_chosen_given_ee / _full = per-candidate P(argmax) by tail kind (1/8 = chance) and their ratio rr with "
        "a group-bootstrap 95% CI. log_coverage as in tail_audit_by_iter (GRPO_LA5 iters 1-2 are partial logs). "
        "GRPO_LA5 ends at iteration 5. Grader: training oracle (gpt-4o-mini)."))
    C.save_table(rt_curve, f"{SCRIPT}_score_by_realized_turns", nd=3, caption=(
        "Mean within-group score deviation (score - group mean; Q1Q2 points) and P(argmax) by realized_turns, per "
        "LA5 arm and train_iter (+ pooled); bootstrap 95% CI over candidates. Odd realized_turns = the tail ends on a "
        "patient turn (the patient closed the session), even = ends on a therapist turn (0 = no tail). Grader: "
        "training oracle (gpt-4o-mini); GRPO_LA5 ends at iteration 5."))

    # ════════════════════════════════════════════════════════════════════════
    # Part 3 — API-call accounting for ALL FOUR arms
    # ════════════════════════════════════════════════════════════════════════
    rec_stats = pd.concat([stream_record_stats(a) for a in arms], ignore_index=True)
    ev = pd.concat([eval_conv_stats(a) for a in arms], ignore_index=True)
    # cross-check vs the tracked session_end_reasons table (pooled over model_iters per arm)
    ser = ev.groupby("arm")[["eval_ended_by_patient", "eval_ended_by_therapist", "eval_ended_none"]].sum()
    print("eval-conv session-end reasons (pooled):\n", ser)

    api_rows = []
    for a in arms:
        steps = grpo_steps(a)
        rs = rec_stats[rec_stats["arm"] == a.label]
        e = ev[ev["arm"] == a.label].set_index("model_iter")
        n_iter = int(rs["train_iter"].max()) if len(rs) else 0
        for n in range(1, n_iter + 2):
            r_train = rs[(rs["train_iter"] == n) & (rs["phase"] != "eval")]
            r_evalph = rs[(rs["train_iter"] == n) & (rs["phase"] == "eval")]
            k = n - 1                                     # eval convs generated at the start of iteration n
            erow = e.loc[k] if k in e.index else None
            final_pass = (n == n_iter + 1)
            n_rec_logged = int(r_train["n_records"].sum()); n_c_logged = int(r_train["n_cands"].sum())
            n_steps = steps.get(n, np.nan) if a.method == "GRPO" else np.nan
            cov = (n_rec_logged / (GRPO_GROUPS_PER_STEP * n_steps)) if (a.method == "GRPO" and n_steps and not final_pass) else 1.0
            cov = cov if cov > 0 else 1.0
            n_rec = int(round(n_rec_logged / cov)); n_c = int(round(n_c_logged / cov))
            n_c_eval = int(round(r_evalph["n_cands"].sum() / cov))
            oc_train = int(round((r_train["oracle_calls"].sum() + r_evalph["oracle_calls"].sum()) / cov))
            pat_tail = int(round((r_train["patient_calls_tail"].sum() + r_evalph["patient_calls_tail"].sum()) / cov))
            # PTO greedy: the trunk's patient reply after each branch point (≤ 1 per record; the last branch
            # point of a trunk that reaches its length cap needs none) — an upper bound within 96 per iteration.
            pat_trunk = n_rec if a.method == "PTO" else 0
            pat_eval = int(erow["eval_patient_turns"]) if erow is not None else 0
            row = {"arm": a.label, "method": a.method, "K": a.K, "train_iter": n,
                   "row_kind": "final eval pass" if final_pass else "iteration",
                   "eval_convs_of": f"model_iter_{k}",
                   "n_eval_convs": int(erow["n_convs"]) if erow is not None else 0,
                   "eval_conv_len_mean": float(erow["eval_conv_len_mean"]) if erow is not None else np.nan,
                   "eval_ended_by_patient": int(erow["eval_ended_by_patient"]) if erow is not None else 0,
                   "n_steps": n_steps, "n_groups_logged": n_rec_logged, "log_coverage": cov,
                   "n_groups": n_rec, "n_candidates": n_c, "n_candidates_eval_phase": n_c_eval,
                   "oracle_calls_train": oc_train,
                   "oracle_retries": int(round((r_train["oracle_retries"].sum() + r_evalph["oracle_retries"].sum()) / cov)),
                   "oracle_input_Mchars": (r_train["oracle_input_chars"].sum() + r_evalph["oracle_input_chars"].sum()) / cov / 1e6,
                   "eval_scoring_calls_run_eval": (int(erow["n_convs"]) * N_EVAL_RUBRICS) if erow is not None else 0,
                   "patient_calls_eval_convs": pat_eval,
                   "patient_calls_trunk": pat_trunk,
                   "patient_calls_tail": pat_tail,
                   "patient_calls_total": pat_eval + pat_trunk + pat_tail,
                   "therapist_gens_tail": int(round((r_train["therapist_gens_tail"].sum() + r_evalph["therapist_gens_tail"].sum()) / cov))}
            api_rows.append(row)
    api = pd.DataFrame(api_rows)
    api = api[~((api["row_kind"] == "final eval pass") & (api["n_eval_convs"] == 0))]  # drop a missing final pass
    C.save_table(api, f"{SCRIPT}_api_calls", nd=2, caption=(
        "API-call accounting per arm x training iteration n (row = the cost of iteration n: the 96 eval convs "
        "generated at its start by policy pi_{n-1} = model_iter_{n-1}, PLUS the training-time calls; the last row "
        "'final eval pass' is the post-loop generate-only pass with no training). oracle_calls_train = Q1 + Q2 "
        "calls per scored candidate (2 per candidate, + recorded retries; GRPO's TRL eval-phase groups included, "
        "n_candidates_eval_phase), read from generations.jsonl and — for GRPO — rescaled to the ground-truth step "
        "count (n_steps = training/completions/*.parquet files, 128 candidates = 16 groups x G=8 per step; "
        "log_coverage = logged / expected groups, < 1 where a crashed iteration's pre-resume records were lost: "
        "GRPO_LA5 iters 1-2, GRPO_LA0 iters 2, 6, 8); oracle_input_Mchars = the chars the oracle read "
        "(prefix + completion + tail, x2 rubrics) as a token proxy. eval_scoring_calls_run_eval = 96 x 8 "
        "instruments per model state (Run_Eval, identical for every arm; per grader). patient_calls_eval_convs = "
        "patient turns in the model_iter_{n-1} CSVs; patient_calls_trunk = PTO greedy trunk replies (<= 1 per "
        "branch point, upper bound within 96); patient_calls_tail = realized patient turns inside K=5 tails "
        "(ceil(realized_turns/2); 1 for a zero-turn tail whose first patient call was made). therapist_gens_tail = "
        "therapist turns generated inside tails (GPU, not API). GRPO_LA5 is right-censored at iteration 5."))

    # K5/K0 ratios per method with arithmetic (sum over matched iterations 1..5, and full run where available)
    ratio_rows = []
    for m in ("PTO", "GRPO"):
        a0 = api[(api["method"] == m) & (api["K"] == 0) & (api["row_kind"] == "iteration")]
        a5 = api[(api["method"] == m) & (api["K"] == 5) & (api["row_kind"] == "iteration")]
        common = sorted(set(a0["train_iter"]) & set(a5["train_iter"]))
        for label, its in (("iters 1-5 (matched)", [i for i in common if i <= 5]), ("all matched iters", common)):
            s0 = a0[a0["train_iter"].isin(its)]; s5 = a5[a5["train_iter"].isin(its)]
            for col in ("oracle_calls_train", "oracle_input_Mchars", "patient_calls_total", "patient_calls_tail", "n_candidates", "total_api_calls"):
                if col == "total_api_calls":
                    v0 = float(s0["oracle_calls_train"].sum() + s0["patient_calls_total"].sum())
                    v5 = float(s5["oracle_calls_train"].sum() + s5["patient_calls_total"].sum())
                else:
                    v0 = float(s0[col].sum()); v5 = float(s5[col].sum())
                ratio_rows.append({"method": m, "window": label, "iters": ",".join(map(str, its)), "quantity": col,
                                   "K0_sum": v0, "K5_sum": v5, "K5_over_K0": (v5 / v0) if v0 else np.nan,
                                   "arithmetic": f"{v5:,.0f} / {v0:,.0f}" if col != "oracle_input_Mchars" else f"{v5:,.1f} / {v0:,.1f}"})
        # per-candidate patient calls under K=5 (the physics: 3 patient calls per full tail)
        n5 = float(a5["n_candidates"].sum()); pt5 = float(a5["patient_calls_tail"].sum())
        ratio_rows.append({"method": m, "window": "all K5 iters", "iters": ",".join(map(str, sorted(a5["train_iter"]))),
                           "quantity": "patient_calls_tail_per_candidate", "K0_sum": 0.0, "K5_sum": pt5 / n5 if n5 else np.nan,
                           "K5_over_K0": np.nan, "arithmetic": f"{pt5:,.0f} / {n5:,.0f} candidates"})
    ratio = pd.DataFrame(ratio_rows)
    C.save_table(ratio, f"{SCRIPT}_api_ratio", nd=3, caption=(
        "K=5 / K=0 API-call ratios per method, summed over the matched training iterations named in 'iters' "
        "(GRPO_LA5 only has iterations 1-5, so the matched window for GRPO is 1-5; PTO also gets 1-10). Sums are "
        "the columns of tail_audit_api_calls.md over those rows (arithmetic shown; total_api_calls = oracle_calls_train + "
        "patient_calls_total). Note the K5/K0 ratio for the "
        "oracle is NOT 1 even though calls per candidate are matched: the number of candidates per iteration "
        "differs between arms (GRPO's prompt count follows the eval-conv length of the current policy; PTO's "
        "branch points follow how far its trunks grow before the patient closes the session)."))

    # ════════════════════════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════════════════════════
    ks = C.K_STYLE[5]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    # (a) ended-early rate by iteration
    ax = axes[0]
    for arm_l in ("PTO_LA5", "GRPO_LA5"):
        d = by_iter[(by_iter["arm"] == arm_l) & (by_iter["train_iter"] != "pooled")].copy()
        d["train_iter"] = d["train_iter"].astype(int)
        ax.plot(d["train_iter"], d["ended_early_rate"], color=pal[arm_l], ls=ks["ls"], marker=ks["marker"], lw=1.6, ms=5, label=arm_l)
        ax.fill_between(d["train_iter"], d["ended_early_ci_lo"], d["ended_early_ci_hi"], color=pal[arm_l], alpha=0.15, lw=0)
        ax.plot(d["train_iter"], d["patient_closed_share"], color=pal[arm_l], ls=":", lw=1.2, alpha=0.9,
                label=f"{arm_l}: patient closed")
    ax.set_xlabel("training iteration"); ax.set_ylabel("share of K=5 tails")
    ax.set_title("(a) tail ended early (<5 turns)", fontsize=8.5)
    ax.set_ylim(0, 0.47); ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=1); ax.set_xticks(range(1, 11))
    ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
    # (b) within-group score deviation by realized_turns (pooled)
    ax = axes[1]
    for arm_l in ("PTO_LA5", "GRPO_LA5"):
        d = rt_curve[(rt_curve["arm"] == arm_l) & (rt_curve["train_iter"] == "pooled")]
        ax.errorbar(d["realized_turns"], d["dev_mean"], yerr=[d["dev_mean"] - d["dev_lo"], d["dev_hi"] - d["dev_mean"]],
                    color=pal[arm_l], ls=ks["ls"], marker=ks["marker"], lw=1.6, ms=5, capsize=2, label=arm_l)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("realized look-ahead turns in tail"); ax.set_ylabel("score − group mean (Q1Q2 pts)")
    ax.set_title("(b) within-group reward by ending", fontsize=8.5)
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(["0\nno tail", "1\npatient\nclosed", "2\nther.\nend", "3\npatient\nclosed", "4\nther.\nend", "5\nfull"],
                       fontsize=6.5)
    ax.set_ylim(-0.15, 0.06); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.tick_params(axis="y", labelsize=8); ax.grid(True, alpha=0.3)
    # (c) P(chosen | ended early) vs P(chosen | full) by iteration
    ax = axes[2]
    for arm_l in ("PTO_LA5", "GRPO_LA5"):
        d = wg[(wg["arm"] == arm_l) & (wg["train_iter"] != "pooled")].copy()
        d["train_iter"] = d["train_iter"].astype(int)
        ax.plot(d["train_iter"], d["p_chosen_given_ee"], color=pal[arm_l], ls=ks["ls"], marker=ks["marker"], lw=1.6, ms=5,
                label=f"{arm_l}: ended early")
        ax.plot(d["train_iter"], d["p_chosen_given_full"], color=pal[arm_l], ls="-", marker="o", mfc="white", lw=1.2, ms=4.5,
                label=f"{arm_l}: full tail")
    ax.axhline(1 / 8, color="grey", lw=0.8, ls=":"); ax.text(10.4, 1 / 8, "1/8", fontsize=7, va="center", ha="right", color="grey")
    ax.set_xlabel("training iteration"); ax.set_ylabel("P(candidate is group argmax)")
    ax.set_title("(c) P(argmax) by tail ending", fontsize=8.5)
    ax.set_ylim(0, None); ax.set_xticks(range(1, 11)); ax.legend(fontsize=6, frameon=False, ncol=1); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Look-ahead tails (K=5 arms) — grader: training oracle (gpt-4o-mini); GRPO_LA5 censored at iteration 5",
                 fontsize=8.5, y=1.02)
    p_fig = C.save_fig(fig, f"{SCRIPT}_fig")

    # API figure
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, col, ttl in ((axes[0], "oracle_calls_train", "(a) oracle calls (training reward)"),
                         (axes[1], "patient_calls_total", "(b) patient-simulator calls")):
        for arm_l in C.ARMS:
            d = api[(api["arm"] == arm_l) & (api["row_kind"] == "iteration")]
            st = C.K_STYLE[C.k_of(arm_l)]
            ax.plot(d["train_iter"], d[col], color=pal[arm_l], ls=st["ls"], marker=st["marker"], lw=1.6, ms=5, label=arm_l)
        ax.set_yscale("log"); ax.set_xlabel("training iteration"); ax.set_ylabel("API calls per iteration (log)")
        ax.set_title(ttl, fontsize=9); ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=8); ax.grid(True, which="both", alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("API calls per training iteration = eval convs of pi_{n-1} + training-time calls "
                 "(GRPO rescaled to its true step count); GRPO_LA5 censored at iteration 5", fontsize=8.5, y=1.02)
    p_fig_api = C.save_fig(fig, f"{SCRIPT}_fig_api")

    # ════════════════════════════════════════════════════════════════════════
    # Ledger
    # ════════════════════════════════════════════════════════════════════════
    L.put("scout_check.GRPO_LA5.iter5.first300groups", scout_check,
          source="tail_audit.py stdout (recomputed from generations.jsonl); scout expected rt {5:1743,3:233,1:374,0:45,4:3,2:2}, ended_early 657/2400=27.4%")
    for _, r in by_iter.iterrows():
        L.put(f"by_iter.{r['arm']}.{r['train_iter']}",
              {k: r[k] for k in ("n_groups", "n_candidates", "log_coverage", "n_unscored_dropped", "ended_early_rate", "ended_early_ci_lo", "ended_early_ci_hi",
                                 "realized_turns_mean", "patient_closed_share", "therapist_stalled_share", "no_tail_share",
                                 "full_share", "full_closed_at_turn5_share", "wrapup_cue_rate_patient_closed", "wrapup_cue_rate_full_open",
                                 "tail_chars_mean", "tail_loop_rate", "cand_floored_rate")},
              source=f"tables/{SCRIPT}_by_iter.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in wg.iterrows():
        L.put(f"within_group.{r['arm']}.{r['train_iter']}",
              {k: r[k] for k in ("n_groups", "n_groups_both", "delta_ee_minus_full", "dz", "ci_lo", "ci_hi", "p", "p_holm",
                                 "rho_dev_vs_realized_turns", "rho_dev_vs_tail_chars", "z_ee_mean", "z_full_mean",
                                 "base_ee_rate", "p_chosen_is_ee", "chosen_minus_base", "chosen_minus_base_ci_lo", "chosen_minus_base_ci_hi",
                                 "p_chosen_given_ee", "p_chosen_given_full", "rr_chosen_ee_vs_full", "rr_ci_lo", "rr_ci_hi")},
              source=f"tables/{SCRIPT}_within_group.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in cues.iterrows():
        L.put(f"cues.{r['arm']}.{r['train_iter']}",
              {k: r[k] for k in ("cand_q_per_turn", "tail_th_q_per_turn", "cand_affirm_rate", "tail_th_affirm_rate",
                                 "cand_effusive_rate", "tail_th_effusive_rate", "cand_len_chars", "tail_th_turn_len_chars", "tail_loop_rate")},
              source=f"tables/{SCRIPT}_cues_by_iter.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in rt_curve[rt_curve["train_iter"] == "pooled"].iterrows():
        L.put(f"score_by_realized_turns.{r['arm']}.pooled.rt{r['realized_turns']}",
              {k: r[k] for k in ("n", "dev_mean", "dev_lo", "dev_hi", "score_mean", "p_chosen")},
              source=f"tables/{SCRIPT}_score_by_realized_turns.md row arm={r['arm']} train_iter=pooled realized_turns={r['realized_turns']}")
    for _, r in api.iterrows():
        L.put(f"api.{r['arm']}.iter{r['train_iter']}",
              {k: r[k] for k in ("row_kind", "eval_convs_of", "n_steps", "n_groups_logged", "log_coverage", "n_groups", "n_candidates", "n_candidates_eval_phase", "oracle_calls_train",
                                 "oracle_retries", "oracle_input_Mchars", "patient_calls_eval_convs", "patient_calls_trunk",
                                 "patient_calls_tail", "patient_calls_total", "therapist_gens_tail", "eval_conv_len_mean", "eval_ended_by_patient")},
              source=f"tables/{SCRIPT}_api_calls.md row arm={r['arm']} train_iter={r['train_iter']}")
    for _, r in ratio.iterrows():
        L.put(f"api_ratio.{r['method']}.{r['window'].replace(' ', '_')}.{r['quantity']}",
              {k: r[k] for k in ("iters", "K0_sum", "K5_sum", "K5_over_K0", "arithmetic")},
              source=f"tables/{SCRIPT}_api_ratio.md row method={r['method']} window={r['window']} quantity={r['quantity']}")
    # arm totals
    for arm_l in C.ARMS:
        d = api[api["arm"] == arm_l]
        L.put(f"api_totals.{arm_l}", {"iters": int((d["row_kind"] == "iteration").sum()),
                                      "oracle_calls_train": int(d["oracle_calls_train"].sum()),
                                      "oracle_input_Mchars": float(d["oracle_input_Mchars"].sum()),
                                      "patient_calls_total": int(d["patient_calls_total"].sum()),
                                      "n_candidates": int(d["n_candidates"].sum()) + int(d["n_candidates_eval_phase"].sum()),
                                      "arithmetic": "column sums over all rows of the arm incl. the final eval pass"},
              source=f"tables/{SCRIPT}_api_calls.md column sums for arm={arm_l}")
    L.put("eval_session_end_reasons_pooled", {a: {c: int(ser.loc[a, c]) for c in ser.columns} for a in ser.index},
          source="tail_audit.py stdout; cross-checks eda/results/L5/tables/3_validity/gpt-4o-mini/session_end_reasons.md")
    L.put("figures", {"fig": str(p_fig), "fig_api": str(p_fig_api)}, source="figures/")
    L.save()
    print("done:", p_fig, p_fig_api)


if __name__ == "__main__":
    main()
