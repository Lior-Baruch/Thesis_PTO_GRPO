"""crossgen.py — every number in the look-ahead chapter, on one measurement axis.

WHAT THIS PRODUCES
------------------
Tables (``../tables/*.md`` + ``.csv``) and figures (``../figures/*.png``) for the
three-generation look-ahead comparison. Nothing in the chapter is hand-computed; each
claim in ``../NUMBERS.md`` names the artifact emitted here.

THE ONE MEASUREMENT AXIS
------------------------
All three generations are reported under the *same* grader: ``gpt-4o-mini-2024-07-18``
with the V5 JSON-schema Q1/Q2 rubric.

* **Exp2 and Exp3** already used exactly that grader — verified byte-identical prompts,
  schemas and labels between ``Exp2_PTO/code/questionnaires.py`` and
  ``Exp3_PTO_GRPO/code/questionnaires.py``.
* **Exp1** was graded by GPT-3.5 with a regex-parsed V1 rubric, so its conversations were
  re-scored by ``eda/tools/score_crossgen.py`` into the ``_crossgen`` partition. Exp1's
  ORIGINAL GPT-3.5 scores are also loaded, so the chapter can show that the look-ahead
  effect there is not an artifact of the old judge.

PAIRING
-------
Every contrast is persona-paired. Exp1 and Exp2 wrote ``conversation_{i}.csv`` with a
FIXED persona order (verified: ``conversation_0`` is the same patient across both K arms
and every iteration), so the file index *is* the persona there. Exp3 reshuffles the 96
personas each iteration (``seed+k+1``), so its pairing goes through
``eda_analysis``'s ``attach_personas`` via ``cross_k_scores`` — a file-index join would
pair unrelated conversations and corrupt every ``dz``.

SIGN CONVENTION
---------------
``Δ = K0 − K5`` throughout, matching the Exp3 artifacts. **A negative Δ means look-ahead
HELPED.** (The ICLR paper reported the opposite orientation; conversions are done here,
never in the prose.)

USAGE
-----
    python analysis/crossgen.py            # tables + figures
    python analysis/crossgen.py --no-figs  # tables only
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PAPER))
EDA = os.path.join(REPO, "Exp3_PTO_GRPO", "eda")
if EDA not in sys.path:
    sys.path.insert(0, EDA)

TABLES = os.path.join(PAPER, "tables")
FIGURES = os.path.join(PAPER, "figures")

EXP1_CONV = os.path.join(REPO, "Exp1_ICLR2025", "data", "conversations_eval")
EXP1_ARM = "TTree1.4_TT0.9_TP0.7_TE0.2_V{}"
EXP1_REGRADED = os.path.join(REPO, "Exp3_PTO_GRPO", "data", "eval_scores", "_crossgen",
                             "judge=openai_gpt-4o-mini-2024-07-18", "rep=0")
EXP2_EVAL = os.path.join(REPO, "Exp2_PTO", "eda", "eval")

PRIMARY = "gpt-4o-mini"
HELDOUT_JUDGE = "anthropic_claude-haiku-4-5"   # Exp3's decoupled second grader

# Which Exp3 optimizers this chapter reports. PTO only: Exp1 and Exp2 are PTO-only, so
# restricting Exp3 to PTO makes look-ahead the single lever varying across all three
# generations. Exp3's GRPO K=5 arm has one matched iteration — too thin to carry a claim —
# and the PTO-vs-GRPO comparison belongs to the companion paper at matched K=0.
EXP3_METHODS = ("PTO",)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                             SHARED STATISTICS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def paired(a: pd.Series, b: pd.Series) -> dict:
    """Persona-paired K0-vs-K5 contrast. Δ = K0 − K5; negative ⇒ look-ahead helped."""
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].values.astype(float), b.loc[idx].values.astype(float)
    d = x - y
    sd = d.std(ddof=1)
    try:
        p = stats.wilcoxon(x, y).pvalue if len(idx) > 1 else np.nan
    except Exception:                       # all-zero differences => no test defined
        p = 1.0
    return dict(n=len(idx), k0=x.mean(), k5=y.mean(), delta=d.mean(),
                dz=(d.mean() / sd if sd > 0 else np.nan), p=p)


def holm(ps: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment within a family."""
    ps = np.asarray(ps, dtype=float)
    ok = ~np.isnan(ps)
    adj = np.full(ps.shape, np.nan)
    vals = ps[ok]
    order = np.argsort(vals)
    m = len(vals)
    run, out = 0.0, np.empty(m)
    for i, j in enumerate(order):
        run = max(run, (m - i) * vals[j])
        out[j] = min(run, 1.0)
    adj[ok] = out
    return adj


def add_holm(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].values)
    return df


def slope(xs, ys) -> float:
    """OLS slope of ys on xs (per-iteration learning rate)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~np.isnan(ys)
    return float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() > 1 else np.nan


def write(df: pd.DataFrame, name: str, note: str = "") -> pd.DataFrame:
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    with open(os.path.join(TABLES, f"{name}.md"), "w", encoding="utf-8") as f:
        if note:
            f.write(f"<!-- {note} -->\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(f"  -> tables/{name}.md")
    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    GENERATION 1 — Exp1 (ICLR 2025)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def exp1_original(k: int, it: int) -> pd.DataFrame:
    """Exp1's ORIGINAL GPT-3.5 / V1-regex scores, keyed by persona (= file index)."""
    d = os.path.join(EXP1_CONV, f"LookAhead_{k}", EXP1_ARM.format(it))
    rows = {}
    for f in glob.glob(os.path.join(d, "scores_*.csv")):
        pid = int(re.search(r"scores_(\d+)", os.path.basename(f)).group(1))
        try:
            df = pd.read_csv(f)
            q1 = pd.to_numeric(df["scores1_avg"].iloc[0], errors="coerce")
            q2 = pd.to_numeric(df["scores2_avg"].iloc[0], errors="coerce")
        except Exception:
            continue
        if pd.notna(q1) and pd.notna(q2):
            rows[pid] = {"Q1": float(q1), "Q2": float(q2), "Q1Q2": (float(q1) + float(q2)) / 2}
    return pd.DataFrame(rows).T.sort_index()


def exp1_regraded(model: str) -> pd.DataFrame:
    """Exp1 re-scored on the Exp3 axis (gpt-4o-mini + V5 Q1/Q2)."""
    cols = {}
    for metric, col in (("Q1", "Q1_Mean"), ("Q2", "Q2_Mean")):
        d = os.path.join(EXP1_REGRADED, f"metric={metric}", "oracle=Q1Q2", model)
        s = {}
        for f in glob.glob(os.path.join(d, "*.csv")):
            try:
                s[int(os.path.splitext(os.path.basename(f))[0])] = float(pd.read_csv(f)[col].iloc[0])
            except Exception:
                pass
        cols[metric] = pd.Series(s, dtype=float)
    df = pd.DataFrame(cols).dropna()
    if len(df):
        df["Q1Q2"] = (df["Q1"] + df["Q2"]) / 2
    return df.sort_index()


def verify_shared_axis() -> dict:
    """Assert Exp2 and Exp3 grade Q1/Q2 identically — the premise of section 4.1.

    The chapter claims generations 2 and 3 need no re-scoring because they already share a
    measurement axis. That is only true while both ``questionnaires.py`` modules build the
    same prompt, schema and labels. Verified here rather than assumed, so a future edit to
    either module fails loudly instead of silently invalidating the comparison.
    """
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    a = load(os.path.join(REPO, "Exp2_PTO", "code", "questionnaires.py"), "_q_exp2")
    b = load(os.path.join(REPO, "Exp3_PTO_GRPO", "code", "questionnaires.py"), "_q_exp3")
    conv = "[THERAPIST]: hello\n[PATIENT]: hi"
    out = {}
    for q in ("Q1", "Q2"):
        pa = a.get_prompt_eval_questionnaire(questionnaire=getattr(a.QuestionnaireID, q),
                                             conversation=conv)
        pb = b.get_prompt_eval_questionnaire(questionnaire=getattr(b.QuestionnaireID, q),
                                             conversation=conv)
        assert pa["prompt"] == pb["prompt"], f"{q}: Exp2/Exp3 prompts have DIVERGED"
        assert pa["schema"] == pb["schema"], f"{q}: Exp2/Exp3 schemas have DIVERGED"
        assert pa["labels"] == pb["labels"], f"{q}: Exp2/Exp3 labels have DIVERGED"
        out[q] = len(pa["prompt"])
    return out


def exp1_grader_agreement() -> pd.DataFrame:
    """How much do the two graders agree on generation 1's SAME conversations?

    Backs the r = 0.774 / +0.269 claims in section 5.3 with a tracked artifact.
    """
    rows = []
    for it in range(1, 8):
        for k in (0, 5):
            old = exp1_original(k, it)["Q1Q2"]
            new = exp1_regraded(f"Exp1_LA{k}_I{it}")["Q1Q2"]
            idx = old.index.intersection(new.index)
            if len(idx) < 3:
                continue
            rows.append(dict(iteration=it, K=k, n=len(idx),
                             gpt35=old.loc[idx].mean(), mini=new.loc[idx].mean(),
                             offset=new.loc[idx].mean() - old.loc[idx].mean(),
                             pearson_r=float(np.corrcoef(old.loc[idx], new.loc[idx])[0, 1]),
                             spearman_rho=float(stats.spearmanr(old.loc[idx],
                                                                new.loc[idx]).statistic)))
    return pd.DataFrame(rows)


def build_exp1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired K contrast under BOTH graders + the per-iteration levels."""
    rows, levels = [], []
    base_old, base_new = 3.453, exp1_regraded("Exp1_Base")["Q1Q2"].mean()
    for it in range(1, 8):
        o0, o5 = exp1_original(0, it), exp1_original(5, it)
        n0, n5 = exp1_regraded(f"Exp1_LA0_I{it}"), exp1_regraded(f"Exp1_LA5_I{it}")
        for metric in ("Q1Q2", "Q1", "Q2"):
            for grader, a, b in ((f"GPT-3.5 (original)", o0[metric], o5[metric]),
                                 (f"{PRIMARY} (re-graded)", n0[metric], n5[metric])):
                rows.append(dict(grader=grader, iteration=it, metric=metric, **paired(a, b)))
        levels.append(dict(iteration=it,
                           gpt35_k0=o0["Q1Q2"].mean(), gpt35_k5=o5["Q1Q2"].mean(),
                           mini_k0=n0["Q1Q2"].mean(), mini_k5=n5["Q1Q2"].mean()))
    lv = pd.DataFrame(levels)
    lv.attrs["base_old"], lv.attrs["base_new"] = base_old, base_new
    return add_holm(pd.DataFrame(rows), ["grader", "iteration"]), lv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         GENERATION 2 — Exp2                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_EXP2_METRIC_COL = {"Q1": ("Q1", "Q1_Mean"), "Q2": ("Q2", "Q2_Mean")}


def exp2_scores(model: str) -> pd.DataFrame:
    """Exp2 per-conversation Q1/Q2 — already on the Exp3 axis (same grader + rubric)."""
    cols = {}
    for metric, (d, col) in _EXP2_METRIC_COL.items():
        s = {}
        for f in glob.glob(os.path.join(EXP2_EVAL, d, model, "*.csv")):
            try:
                s[int(os.path.splitext(os.path.basename(f))[0])] = float(pd.read_csv(f)[col].iloc[0])
            except Exception:
                pass
        cols[metric] = pd.Series(s, dtype=float)
    df = pd.DataFrame(cols).dropna()
    if len(df):
        df["Q1Q2"] = (df["Q1"] + df["Q2"]) / 2
    return df.sort_index()


# Iteration coverage differs per training oracle; only matched iterations enter the contrast.
EXP2_ORACLES = {"Q1Q2": range(1, 6), "WAI": range(1, 6), "CSQ8": range(1, 6)}

# Exp2 was scored on six questionnaires, so its K contrast can be checked far more widely
# than Q1+Q2. Folder -> the per-conversation summary column.
EXP2_ALL_RUBRICS = {
    "Q1": ("Q1", "Q1_Mean"), "Q2": ("Q2", "Q2_Mean"),
    "WAI-SR": ("WAI_SR", "WAI_TotalMean"), "CSQ-8": ("CSQ8", "CSQ8_Mean"),
    "MI-SAT": ("MI_SAT", "MI_Mean"), "MITI": ("MITI", "MITI_GlobalMean"),
}


def exp2_rubric(model: str, rubric: str) -> pd.Series:
    d, col = EXP2_ALL_RUBRICS[rubric]
    s = {}
    for f in glob.glob(os.path.join(EXP2_EVAL, d, model, "*.csv")):
        try:
            s[int(os.path.splitext(os.path.basename(f))[0])] = float(pd.read_csv(f)[col].iloc[0])
        except Exception:
            pass
    return pd.Series(s, dtype=float).sort_index()


def build_exp2_all_rubrics() -> pd.DataFrame:
    """The widest available Exp2 K contrast: 3 oracles x 5 iters x 7 metrics."""
    rows = []
    for oracle, iters in EXP2_ORACLES.items():
        for it in iters:
            m0, m5 = f"L0_{oracle}_V{it}", f"L5_{oracle}_V{it}"
            comp0, comp5 = exp2_scores(m0), exp2_scores(m5)
            if len(comp0) and len(comp5):
                rows.append(dict(oracle=oracle, iteration=it, metric="Q1Q2",
                                 **paired(comp0["Q1Q2"], comp5["Q1Q2"])))
            for rubric in EXP2_ALL_RUBRICS:
                a, b = exp2_rubric(m0, rubric), exp2_rubric(m5, rubric)
                if len(a) and len(b):
                    rows.append(dict(oracle=oracle, iteration=it, metric=rubric, **paired(a, b)))
    return add_holm(pd.DataFrame(rows), ["oracle", "iteration"])


def build_exp2() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, levels = [], []
    for oracle, iters in EXP2_ORACLES.items():
        for it in iters:
            s0, s5 = exp2_scores(f"L0_{oracle}_V{it}"), exp2_scores(f"L5_{oracle}_V{it}")
            if not len(s0) or not len(s5):
                continue
            for metric in ("Q1Q2", "Q1", "Q2"):
                rows.append(dict(oracle=oracle, iteration=it, metric=metric,
                                 **paired(s0[metric], s5[metric])))
            levels.append(dict(oracle=oracle, iteration=it,
                               k0=s0["Q1Q2"].mean(), k5=s5["Q1Q2"].mean()))
    return add_holm(pd.DataFrame(rows), ["oracle", "iteration"]), pd.DataFrame(levels)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         GENERATION 3 — Exp3                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_exp3(judge: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp3's cross-K frame, persona-paired by ``attach_personas`` (NOT by file index).

    ``judge`` selects the grader: "" = the primary oracle, or a judge tag such as
    ``anthropic_claude-haiku-4-5`` for the held-out grader. Both graders scored the full
    grid, so the same contrast can be recomputed under either.
    """
    from eda_analysis.config import EdaConfig, notebook_setup, cross_k_scores

    S = notebook_setup(EdaConfig(view="L5", judge=judge) if judge else EdaConfig(view="L5"))
    df = cross_k_scores(S)
    df = df[df.questionnaire.isin(["Q1Q2", "Q1", "Q2"])]

    rows, levels = [], []
    for method in sorted(set(df.method.unique()) & set(EXP3_METHODS)):
        m = df[df.method == method]
        common = sorted(set(m[m.K == 0].iteration) & set(m[m.K == 5].iteration) - {0})
        for it in common:
            for metric in ("Q1Q2", "Q1", "Q2"):
                cell = m[(m.iteration == it) & (m.questionnaire == metric)]
                a = cell[cell.K == 0].set_index("persona_id")["score"]
                b = cell[cell.K == 5].set_index("persona_id")["score"]
                if a.index.duplicated().any() or b.index.duplicated().any():
                    a, b = a.groupby(level=0).mean(), b.groupby(level=0).mean()
                rows.append(dict(method=method, iteration=it, metric=metric, **paired(a, b)))
            cell = m[(m.iteration == it) & (m.questionnaire == "Q1Q2")]
            levels.append(dict(method=method, iteration=it,
                               k0=cell[cell.K == 0]["score"].mean(),
                               k5=cell[cell.K == 5]["score"].mean()))
    # base level (iteration 0, shared by both K arms)
    b0 = df[(df.iteration == 0) & (df.questionnaire == "Q1Q2")]["score"].mean()
    lv = pd.DataFrame(levels)
    lv.attrs["base"] = b0
    return add_holm(pd.DataFrame(rows), ["method", "iteration"]), lv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        THE MODERATOR ANALYSIS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_moderator(e1: pd.DataFrame, e1lv: pd.DataFrame,
                    e2: pd.DataFrame, e2lv: pd.DataFrame,
                    e3: pd.DataFrame, e3lv: pd.DataFrame) -> pd.DataFrame:
    """Does look-ahead's benefit track how badly the MYOPIC arm is doing?

    One row per training arm. ``myopic_slope`` is the OLS per-iteration slope of the K=0
    arm's Q1+Q2; ``myopic_gain`` is its endpoint minus base. ``la_benefit`` is the mean
    K=5 advantage over matched iterations (= −mean Δ, so POSITIVE means look-ahead helped).
    """
    rows = []

    lv = e1lv
    k0 = lv["mini_k0"].values
    rows.append(dict(
        generation="Exp1 (ICLR'25)", arm="PTO / Q1Q2", therapist="Llama-2-7B",
        patient="GPT-3.5 (cooperative)", n_iters=len(lv),
        base=lv.attrs["base_new"], myopic_end=k0[-1],
        myopic_gain=k0[-1] - lv.attrs["base_new"],
        myopic_slope=slope(lv["iteration"], k0),
        la_benefit=-e1[(e1.grader.str.startswith(PRIMARY)) & (e1.metric == "Q1Q2")]["delta"].mean()))

    for oracle in EXP2_ORACLES:
        sub = e2lv[e2lv.oracle == oracle]
        if not len(sub):
            continue
        base = exp2_scores("Base")["Q1Q2"].mean()
        rows.append(dict(
            generation="Exp2", arm=f"PTO / {oracle}", therapist="Llama-3.2-1B (4-bit)",
            patient="gpt-4o-mini (less coop.)", n_iters=len(sub),
            base=base, myopic_end=sub["k0"].values[-1],
            myopic_gain=sub["k0"].values[-1] - base,
            myopic_slope=slope(sub["iteration"], sub["k0"]),
            la_benefit=-e2[(e2.oracle == oracle) & (e2.metric == "Q1Q2")]["delta"].mean()))

    for method in sorted(e3lv.method.unique()):
        sub = e3lv[e3lv.method == method]
        if not len(sub):
            continue
        rows.append(dict(
            generation="Exp3", arm=f"{method} / Q1Q2", therapist="Llama-3.2-1B (bf16)",
            patient="gpt-4o-mini (less coop.)", n_iters=len(sub),
            base=e3lv.attrs["base"], myopic_end=sub["k0"].values[-1],
            myopic_gain=sub["k0"].values[-1] - e3lv.attrs["base"],
            myopic_slope=slope(sub["iteration"], sub["k0"]),
            la_benefit=-e3[(e3.method == method) & (e3.metric == "Q1Q2")]["delta"].mean()))

    df = pd.DataFrame(rows)
    ok = df["n_iters"] >= 3          # an arm needs >=3 matched iterations to fit a slope
    if ok.sum() >= 3:
        for key, col in (("slope", "myopic_slope"), ("gain", "myopic_gain")):
            r, p = stats.pearsonr(df.loc[ok, col], df.loc[ok, "la_benefit"])
            rho, prho = stats.spearmanr(df.loc[ok, col], df.loc[ok, "la_benefit"])
            df.attrs[f"r_{key}"], df.attrs[f"p_{key}"] = r, p
            df.attrs[f"rho_{key}"], df.attrs[f"prho_{key}"] = rho, prho
        df.attrs["n_used"] = int(ok.sum())
    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                 FIGURES                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def make_figures(e1lv, e2lv, e3lv, mod) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eda_analysis import plotting_style

    plotting_style.set_style()
    os.makedirs(FIGURES, exist_ok=True)
    C0, C5 = "#4C72B0", "#DD8452"          # K=0 / K=5, consistent across every panel

    # ── Figure 1: the three trajectories, one measurement axis ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    ax = axes[0]
    ax.plot(e1lv["iteration"], e1lv["mini_k0"], "o-", color=C0, label="K=0")
    ax.plot(e1lv["iteration"], e1lv["mini_k5"], "s-", color=C5, label="K=5")
    ax.axhline(e1lv.attrs["base_new"], ls=":", color="0.4", lw=1)
    ax.set_title("Exp1 (ICLR'25) — Llama-2-7B\nlook-ahead helps", fontsize=10)
    ax.set_xlabel("iteration"); ax.set_ylabel("Q1+Q2 (gpt-4o-mini)")

    ax = axes[1]
    for oracle, mk in zip(EXP2_ORACLES, ("o", "^", "v")):
        sub = e2lv[e2lv.oracle == oracle]
        ax.plot(sub["iteration"], sub["k0"], mk + "-", color=C0, alpha=.75,
                label="K=0" if oracle == "Q1Q2" else None)
        ax.plot(sub["iteration"], sub["k5"], mk + "--", color=C5, alpha=.75,
                label="K=5" if oracle == "Q1Q2" else None)
    ax.axhline(exp2_scores("Base")["Q1Q2"].mean(), ls=":", color="0.4", lw=1)
    ax.set_title("Exp2 — Llama-3.2-1B (4-bit)\nnull (3 oracles)", fontsize=10)
    ax.set_xlabel("iteration")

    ax = axes[2]
    sub = e3lv[e3lv.method == "PTO"]
    ax.plot(sub["iteration"], sub["k0"], "o-", color=C0, label="K=0")
    ax.plot(sub["iteration"], sub["k5"], "s-", color=C5, label="K=5")
    ax.axhline(e3lv.attrs["base"], ls=":", color="0.4", lw=1)
    ax.set_title("Exp3 — Llama-3.2-1B (bf16)\nlook-ahead never leads", fontsize=10)
    ax.set_xlabel("iteration")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_trajectories.png"), dpi=200)
    plt.close(fig)
    print("  -> figures/fig1_trajectories.png")

    # ── Figure 2: the moderator ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    use = mod[mod["n_iters"] >= 3]
    colors = {"Exp1 (ICLR'25)": "#55A868", "Exp2": "#C44E52", "Exp3": "#8172B3"}
    for _, r in use.iterrows():
        ax.scatter(r["myopic_slope"], r["la_benefit"], s=110,
                   color=colors.get(r["generation"], "0.5"), zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{r['generation'].split()[0]} {r['arm'].split('/')[-1].strip()}",
                    (r["myopic_slope"], r["la_benefit"]), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.margins(x=0.18)                       # keep the right-most arm's label on-canvas
    if len(use) > 2:
        xs = np.linspace(use["myopic_slope"].min(), use["myopic_slope"].max(), 50)
        b, a = np.polyfit(use["myopic_slope"], use["la_benefit"], 1)
        ax.plot(xs, a + b * xs, color="0.35", lw=1.2, zorder=1)
    ax.set_xlabel("myopic (K=0) learning rate — Q1+Q2 per iteration")
    ax.set_ylabel("look-ahead benefit\n(mean K=5 − K=0, Q1+Q2)")
    ttl = "Look-ahead pays off only where the myopic signal fails"
    if "r_slope" in mod.attrs:
        ttl += (f"\nPearson r = {mod.attrs['r_slope']:.2f}, p = {mod.attrs['p_slope']:.2f} "
                f"(n = {mod.attrs['n_used']} arms) — descriptive, not powered")
    ax.set_title(ttl, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_moderator.png"), dpi=200)
    plt.close(fig)
    print("  -> figures/fig2_moderator.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    print("Premise check — do Exp2 and Exp3 grade Q1/Q2 identically?")
    lens = verify_shared_axis()
    print(f"  OK: byte-identical prompts, schemas and labels "
          f"(Q1 {lens['Q1']} chars, Q2 {lens['Q2']} chars) — no re-scoring needed for Exp2")

    print("Generation 1 — Exp1 (ICLR 2025), two graders")
    e1, e1lv = build_exp1()
    agree = write(exp1_grader_agreement(), "t11_exp1_grader_agreement",
                  "Cross-grader agreement on Exp1's SAME conversations: GPT-3.5 (original) vs "
                  "gpt-4o-mini (re-graded), per arm.")
    print(f"     cross-grader r: mean {agree.pearson_r.mean():.3f} "
          f"[{agree.pearson_r.min():.3f}, {agree.pearson_r.max():.3f}], "
          f"level offset {agree.offset.mean():+.3f}")
    write(e1, "t2_exp1_two_graders",
          "Exp1 paired K contrast under the original GPT-3.5 grader and the Exp3 "
          "gpt-4o-mini re-grade. Delta = K0-K5; NEGATIVE means look-ahead helped.")

    print("Generation 2 — Exp2")
    e2, e2lv = build_exp2()
    write(e2, "t3_exp2_k",
          "Exp2 paired K contrast by training oracle. Already on the Exp3 axis "
          "(byte-identical Q1/Q2 prompts + same grader). Delta = K0-K5.")

    e2all = build_exp2_all_rubrics()
    write(e2all, "t9_exp2_all_rubrics",
          "Exp2 K contrast across every rubric it was scored on (3 oracles x 5 iterations x "
          "7 metrics). Delta = K0-K5.")
    sig = int((e2all.p_holm < .05).sum())
    print(f"     all-rubric sweep: {len(e2all)} contrasts, {sig} Holm-significant, "
          f"K=5 ahead in {int((e2all.delta < 0).sum())} ({(e2all.delta < 0).mean():.1%})")

    print("Generation 3 — Exp3")
    e3, e3lv = build_exp3()
    write(e3, "t4_exp3_k",
          "Exp3 paired K contrast by method, persona-matched via attach_personas. "
          "Delta = K0-K5.")

    # The held-out grader (different model family, never played the patient, never touched
    # training) re-scored the same grid, so the contrast recomputes under it unchanged.
    e3h, _ = build_exp3(judge=HELDOUT_JUDGE)
    write(e3h, "t10_exp3_k_heldout",
          f"Exp3 paired K contrast under the HELD-OUT grader ({HELDOUT_JUDGE}). Delta = K0-K5.")
    hp = e3h[(e3h.method == "PTO") & (e3h.metric == "Q1Q2")]
    print(f"     held-out judge, PTO Q1Q2: K=5 ahead {int((hp.delta < 0).sum())}/{len(hp)}, "
          f"mean delta {hp.delta.mean():+.3f}, Holm-sig favouring K=0 "
          f"{int(((hp.p_holm < .05) & (hp.delta > 0)).sum())}")

    # ── the headline cross-generation summary ─────────────────────────────────
    def summarise(label, sub, iters, extra):
        """One headline row. ``matched_contrasts`` counts persona-paired K0-vs-K5 cells on
        Q1+Q2 — for Exp2 that is 3 training oracles x 5 iterations, not 15 iterations."""
        q = sub[sub.metric == "Q1Q2"]
        return dict(generation=label, matched_contrasts=iters,
                    k5_ahead=f"{int((q.delta < 0).sum())}/{len(q)}",
                    mean_delta=q["delta"].mean(),
                    mean_dz=q["dz"].mean(),
                    holm_sig_k5=int(((q.p_holm < .05) & (q.delta < 0)).sum()),
                    holm_sig_k0=int(((q.p_holm < .05) & (q.delta > 0)).sum()),
                    **extra)

    e1m = e1[e1.grader.str.startswith(PRIMARY)]
    e3p = e3[e3.method == "PTO"]
    headline = pd.DataFrame([
        summarise("Exp1 (ICLR'25) — Llama-2-7B", e1m, 7,
                  dict(therapist="Llama-2-7B", patient="GPT-3.5", verdict="look-ahead HELPS")),
        summarise("Exp2 — Llama-3.2-1B (4-bit)", e2, len(e2[e2.metric == "Q1Q2"]),
                  dict(therapist="Llama-3.2-1B 4-bit", patient="gpt-4o-mini", verdict="NULL")),
        summarise("Exp3 — Llama-3.2-1B (bf16), PTO", e3p, len(e3p[e3p.metric == "Q1Q2"]),
                  dict(therapist="Llama-3.2-1B bf16", patient="gpt-4o-mini",
                       verdict="look-ahead NEVER LEADS")),
    ])
    write(headline, "t1_generations",
          "THE HEADLINE. All three generations under ONE grader (gpt-4o-mini + V5 Q1/Q2). "
          "Delta = K0-K5, so negative favours look-ahead.")

    print("Moderator")
    mod = build_moderator(e1, e1lv, e2, e2lv, e3, e3lv)
    write(mod, "t5_moderator",
          "Look-ahead benefit vs how well the MYOPIC arm learns. la_benefit is oriented so "
          "POSITIVE = look-ahead helped.")

    # levels, for the trajectory figure and the appendix
    write(e1lv.assign(base_gpt35=e1lv.attrs["base_old"], base_mini=e1lv.attrs["base_new"]),
          "t6_exp1_levels", "Exp1 Q1+Q2 per iteration under both graders.")
    write(e2lv, "t7_exp2_levels", "Exp2 Q1+Q2 per iteration per training oracle.")
    write(e3lv.assign(base=e3lv.attrs["base"]), "t8_exp3_levels",
          "Exp3 Q1+Q2 per iteration per method.")

    print("\n" + "=" * 92)
    print(headline.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("=" * 92)
    if "r_slope" in mod.attrs:
        print(f"\nModerator (n = {mod.attrs['n_used']} arms) — UNDERPOWERED, descriptive only:")
        for key, lab in (("slope", "myopic slope"), ("gain", "myopic gain ")):
            print(f"  vs {lab}: Pearson r = {mod.attrs['r_' + key]:+.3f} "
                  f"(p = {mod.attrs['p_' + key]:.3f}),  "
                  f"Spearman rho = {mod.attrs['rho_' + key]:+.3f} "
                  f"(p = {mod.attrs['prho_' + key]:.3f})")
    print(mod.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

    if not args.no_figs:
        print("\nFigures")
        make_figures(e1lv, e2lv, e3lv, mod)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
