#!/usr/bin/env python
"""
make_exp1_exp2_figs.py — render the Exp1 and Exp2 figures the deck needs.

Exp3's EDA renders ~200 PNGs under ``Exp3_PTO_GRPO/eda/results/``. **Exp1 and Exp2 render none** —
their EDA notebooks (`Exp1_ICLR2025/eda/Conv_EDA.ipynb`, `Exp2_PTO/eda/Conv_EDA.ipynb`) draw inline
and write nothing to disk, so every Exp1/Exp2 figure ever shown was recomputed in a notebook. This
script produces them as files so a deck can cite an artifact rather than a screenshot.

    Exp1  ← raw per-conversation CSVs, Exp1_ICLR2025/data/conversations_eval/**
          (scores_N.csv: scores1_avg = Q1, scores2_avg = Q2, scores_avg = Final;
           conversation_N.csv: one row per utterance)
    Exp2  ← meetings/build/_exp2_summary.csv, itself rebuilt from Exp2_PTO/eda/eval/**

⚠ **Exp1 and Exp2 are PTO-only.** Exp2's GRPO V1 run had a bug and its scores are void
(`Exp2_PTO/CLAUDE.md` § "GRPO V1 — VOID"); `_exp2_summary.csv` has those states stripped and this
script never re-adds them.

⚠ Exp1's `conversations_eval/` holds 42 run_dirs, only 15 of which are the paper's models. The
mapping below is the paper's Table 1, identified by exact numeric match on all six statistics —
the paper itself never discloses which baseline config it used. Other run_dirs (`_FAIL_Q2`,
`_OLD`, the `Q2_*` and `FullEval_*` sweeps, `LookAhead_3`, `LookAhead_10`) are deliberately NOT
plotted; `LookAhead_3` in particular ran at different hyperparameters and is not a dose-response
point.

Output: meetings/build/_figs/{exp1,exp2}/*.png at 200 dpi.

Run:
    & ..\\..\\.venv\\Scripts\\python.exe make_exp1_exp2_figs.py
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXP1 = os.path.join(REPO, "Exp1_ICLR2025", "data", "conversations_eval")
EXP2_CSV = os.path.join(HERE, "_exp2_summary.csv")
OUT1 = os.path.join(HERE, "_figs", "exp1")
OUT2 = os.path.join(HERE, "_figs", "exp2")

BOOT_SEED = 20260823
N_BOOT = 5000

# Match the Exp3 EDA's arm identities so the deck reads as one system.
K0_C, K5_C = "#1f77b4", "#7fb9dd"
ORACLE_C = {"Q1Q2": "#1f77b4", "WAI": "#8c6bb1", "CSQ8": "#2f8f5b"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})

# ── Exp1: the paper's 15 models ───────────────────────────────────────────────
P = "TTree1.4_TT0.9_TP0.7_TE0.2_V%d"
EXP1_MODELS = [("Base", 0, os.path.join("Base", "Basic_50_TT0.9_TP0.7_TE0.2_V2"))]
EXP1_MODELS += [("K=0", i, os.path.join("LookAhead_0", P % i)) for i in range(1, 8)]
EXP1_MODELS += [("K=5", i, os.path.join("LookAhead_5", P % i)) for i in range(1, 8)]


def _load_exp1():
    """Per-conversation Q1 / Q2 / Final / length for each of the paper's 15 models."""
    rows = []
    for arm, it, rel in EXP1_MODELS:
        d = os.path.join(EXP1, rel)
        for i in range(96):
            sp = os.path.join(d, "scores_%d.csv" % i)
            cp = os.path.join(d, "conversation_%d.csv" % i)
            if not os.path.exists(sp):
                continue
            s = pd.read_csv(sp)
            n_utt = len(pd.read_csv(cp)) if os.path.exists(cp) else np.nan
            rows.append({
                "arm": arm, "iteration": it, "persona": i,
                "Q1": float(s["scores1_avg"].iloc[0]),
                "Q2": float(s["scores2_avg"].iloc[0]),
                "Final": float(s["scores_avg"].iloc[0]),
                "length": n_utt,
            })
    df = pd.DataFrame(rows)
    # One corrupt file in 4,031: Base/Good_50 scores_12 parsed item NUMBERS as scores. That run_dir
    # is not plotted, but guard the whole frame anyway — nothing here is on a 1-5 scale above 5.
    bad = df[["Q1", "Q2", "Final"]].gt(5.0).any(axis=1)
    if bad.any():
        print("  dropped %d rows with an out-of-scale score" % int(bad.sum()))
        df = df[~bad]
    return df


def _boot_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def _paired(df, metric, it):
    """K=0 minus K=5 at one iteration, paired on persona index."""
    a = df[(df.arm == "K=0") & (df.iteration == it)].set_index("persona")[metric]
    b = df[(df.arm == "K=5") & (df.iteration == it)].set_index("persona")[metric]
    j = a.index.intersection(b.index)
    return (a.loc[j] - b.loc[j]).values


# ══════════════════════════════════════════════════════════════════════════════
def exp1_trajectory(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, m in zip(axes, ["Final", "Q1", "Q2"]):
        base = df[df.arm == "Base"][m]
        ax.axhline(base.mean(), ls=":", lw=1.4, color="#888888",
                   label="Base (untrained)" if m == "Final" else None)
        for arm, c, ls, mk in [("K=0", K0_C, "-", "o"), ("K=5", K5_C, "--", "s")]:
            g = df[df.arm == arm].groupby("iteration")[m]
            mu, se = g.mean(), g.sem()
            ax.plot(mu.index, mu.values, ls=ls, marker=mk, color=c, lw=2, ms=6,
                    label="PTO %s" % arm if m == "Final" else None)
            ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.20, lw=0)
        ax.set_title("%s%s" % (m, {"Q1": "  (session satisfaction)",
                                   "Q2": "  (working alliance)"}.get(m, "  (mean of Q1, Q2)")))
        ax.set_xlabel("PTO iteration")
        ax.set_xticks(range(1, 8))
    axes[0].set_ylabel("oracle score (1–5), mean ± SE")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Exp1 — score by PTO iteration, K=0 vs K=5   (GPT-3.5 oracle, n = 96 per point)",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT1, "exp1_trajectory.png"))
    plt.close(fig)


def exp1_paired_delta(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, m in zip(axes, ["Final", "Q1", "Q2"]):
        its, mus, los, his = [], [], [], []
        for it in range(1, 8):
            dd = _paired(df, m, it)
            its.append(it)
            mus.append(dd.mean())
            lo, hi = _boot_ci(dd)
            los.append(lo)
            his.append(hi)
        mus = np.array(mus)
        ax.axhline(0, color="black", lw=1)
        ax.errorbar(its, mus, yerr=[mus - np.array(los), np.array(his) - mus],
                    fmt="o", color=K0_C, ms=7, capsize=3, lw=1.6)
        ax.set_title(m)
        ax.set_xlabel("PTO iteration")
        ax.set_xticks(range(1, 8))
    axes[0].set_ylabel("Δ (K=0 − K=5), 95% CI")
    fig.suptitle("Exp1 — paired K=0 minus K=5 by iteration   "
                 "(same 96 personas; below 0 = K=5 scored higher)",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT1, "exp1_paired_delta.png"))
    plt.close(fig)


def exp1_distributions(df):
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    order, colors, labels = [], [], []
    order.append(df[df.arm == "Base"]["Final"].values)
    colors.append("#999999")
    labels.append("Base")
    for arm, c in [("K=0", K0_C), ("K=5", K5_C)]:
        for it in range(1, 8):
            order.append(df[(df.arm == arm) & (df.iteration == it)]["Final"].values)
            colors.append(c)
            labels.append("%s\nI%d" % (arm, it))
    parts = ax.violinplot(order, showextrema=False, widths=0.85)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        b.set_edgecolor("none")
    bp = ax.boxplot(order, widths=0.16, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Final score (1–5), per conversation")
    ax.set_title("Exp1 — per-conversation Final-score distribution, all 15 published models "
                 "(96 conversations each)", fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT1, "exp1_distributions.png"))
    plt.close(fig)


def exp1_sd_and_length(df):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.1))
    ax = axes[0]
    b = df[df.arm == "Base"]["Final"]
    ax.axhline(b.std(ddof=1), ls=":", lw=1.4, color="#888888", label="Base")
    for arm, c, ls, mk in [("K=0", K0_C, "-", "o"), ("K=5", K5_C, "--", "s")]:
        g = df[df.arm == arm].groupby("iteration")["Final"].std(ddof=1)
        ax.plot(g.index, g.values, ls=ls, marker=mk, color=c, lw=2, ms=6, label="PTO " + arm)
    ax.set_xlabel("PTO iteration")
    ax.set_ylabel("SD of Final score across the 96 conversations")
    ax.set_xticks(range(1, 8))
    ax.set_title("Spread")
    ax.legend(fontsize=9)

    ax = axes[1]
    bl = df[df.arm == "Base"]["length"]
    ax.axhline(bl.mean(), ls=":", lw=1.4, color="#888888", label="Base")
    for arm, c, ls, mk in [("K=0", K0_C, "-", "o"), ("K=5", K5_C, "--", "s")]:
        g = df[df.arm == arm].groupby("iteration")["length"]
        mu, se = g.mean(), g.sem()
        ax.plot(mu.index, mu.values, ls=ls, marker=mk, color=c, lw=2, ms=6, label="PTO " + arm)
        ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.20, lw=0)
    ax.set_xlabel("PTO iteration")
    ax.set_ylabel("utterances per conversation, mean ± SE")
    ax.set_xticks(range(1, 8))
    ax.set_title("Session length")
    ax.legend(fontsize=9)
    fig.suptitle("Exp1 — score spread and session length by iteration   (n = 96 per point)",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT1, "exp1_sd_length.png"))
    plt.close(fig)


def exp1_vs_base(df):
    base = df[df.arm == "Base"].set_index("persona")["Final"]
    ys, mus, los, his, cols = [], [], [], [], []
    for arm, c in [("K=5", K5_C), ("K=0", K0_C)]:
        for it in range(7, 0, -1):
            v = df[(df.arm == arm) & (df.iteration == it)].set_index("persona")["Final"]
            j = v.index.intersection(base.index)
            dd = (v.loc[j] - base.loc[j]).values
            ys.append("%s  I%d" % (arm, it))
            mus.append(dd.mean())
            lo, hi = _boot_ci(dd)
            los.append(lo)
            his.append(hi)
            cols.append(c)
    y = np.arange(len(ys))
    mus = np.array(mus)
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.axvline(0, color="black", lw=1)
    # errorbar's ecolor takes ONE colour, so draw the bars one at a time.
    for yi, mi, lo_i, hi_i, ci in zip(y, mus, los, his, cols):
        ax.plot([lo_i, hi_i], [yi, yi], color=ci, lw=1.6, solid_capstyle="butt", zorder=2)
    ax.scatter(mus, y, color=cols, s=52, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ys, fontsize=9)
    ax.set_xlabel("Δ Final score vs the untrained baseline (paired on persona, 95% CI)")
    ax.set_title("Exp1 — every published model against Base   (n = 96 paired)",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT1, "exp1_vs_base.png"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
def _e2(df, metric):
    return df[df.metric == metric]


def exp2_trajectory(df):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    m = _e2(df, "Q1Q2_Mean")
    base = m[m.model_state == "Base"]["mean"].iloc[0]
    ax.axhline(base, ls=":", lw=1.6, color="#888888")
    ax.annotate("Base (untrained, 4-bit) = %.3f" % base, xy=(10.2, base), fontsize=9,
                color="#666666", va="center")
    for orc in ["Q1Q2", "WAI", "CSQ8"]:
        for K, ls, mk in [(0.0, "-", "o"), (5.0, "--", "s")]:
            g = m[(m.train_oracle == orc) & (m.K == K)].sort_values("iteration")
            if g.empty:
                continue
            ax.errorbar(g["iteration"], g["mean"],
                        yerr=[g["mean"] - g["ci95_lo"], g["ci95_hi"] - g["mean"]],
                        ls=ls, marker=mk, color=ORACLE_C[orc], lw=1.9, ms=6, capsize=2,
                        alpha=0.95, label="%s  K=%d" % (orc, int(K)))
    ax.set_xlabel("PTO iteration (merged DPO adapters)")
    ax.set_ylabel("Q1+Q2 (1–5), mean with 95% CI")
    ax.set_xticks(range(1, 11))
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    ax.set_title("Exp2 — Q1+Q2 by iteration, three training oracles × K   (n = 96 per point)",
                 fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT2, "exp2_trajectory_q1q2.png"))
    plt.close(fig)


def exp2_own_instrument(df):
    """Each arm scored on the instrument it was TRAINED on."""
    pairs = [("Q1Q2", "Q1Q2_Mean", "Q1+Q2"), ("WAI", "WAI_TotalMean", "WAI-SR total"),
             ("CSQ8", "CSQ8_Mean", "CSQ-8")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (orc, metric, nice) in zip(axes, pairs):
        m = _e2(df, metric)
        base = m[m.model_state == "Base"]["mean"].iloc[0]
        ax.axhline(base, ls=":", lw=1.5, color="#888888", label="Base = %.3f" % base)
        for K, ls, mk in [(0.0, "-", "o"), (5.0, "--", "s")]:
            g = m[(m.train_oracle == orc) & (m.K == K)].sort_values("iteration")
            if g.empty:
                continue
            ax.errorbar(g["iteration"], g["mean"],
                        yerr=[g["mean"] - g["ci95_lo"], g["ci95_hi"] - g["mean"]],
                        ls=ls, marker=mk, color=ORACLE_C[orc], lw=2, ms=6, capsize=2,
                        label="K=%d" % int(K))
        ax.set_title("trained on %s → scored on %s" % (orc, nice), fontsize=10.5)
        ax.set_xlabel("PTO iteration")
        ax.set_xticks(range(1, 11 if orc == "Q1Q2" else 6))
        ax.legend(fontsize=8.5, loc="best")
    axes[0].set_ylabel("score on the training instrument, mean ± 95% CI")
    fig.suptitle("Exp2 — each arm measured on its OWN training instrument   (n = 96 per point)",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.savefig(os.path.join(OUT2, "exp2_own_instrument.png"))
    plt.close(fig)


def exp2_vs_base_forest(df):
    """Paired Δ vs Base with a CI ON THE PAIRED DIFFERENCE.

    ⚠ This drew `paired_delta ± 1.96 * sem` until 2026-08-24, where `sem` is the model state's OWN
    unpaired SEM (sd/sqrt(n)). That ignores both the Base arm's variance and the persona-pairing
    correlation, so the bar had nothing to do with the interval around the difference — and because
    the point estimate and the significance marker ARE paired, three rows contradicted themselves
    on the slide (two filled markers whose bars crossed zero, one hollow marker whose bar did not).
    The paired SD is recovered exactly from the tabulated dz: SD(delta) = delta / dz.
    """
    m = _e2(df, "Q1Q2_Mean")
    m = m[m.model_state != "Base"].copy()
    m["key"] = m["train_oracle"] + " K=" + m["K"].astype(int).astype(str) + " I" + \
        m["iteration"].astype(int).astype(str)
    m = m.sort_values(["train_oracle", "K", "iteration"], ascending=[True, True, False])
    y = np.arange(len(m))
    cols = [ORACLE_C[o] for o in m["train_oracle"]]

    d = m["paired_delta_vs_ref"].values.astype(float)
    dz = m["dz_vs_ref"].values.astype(float)
    n = m["n"].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_paired = np.where(np.abs(dz) > 1e-9, d / dz, np.nan)
    se_paired = sd_paired / np.sqrt(n)
    lo, hi = d - 1.96 * se_paired, d + 1.96 * se_paired

    fig, ax = plt.subplots(figsize=(11.0, 8.8))
    ax.axvline(0, color="black", lw=1)
    for yi, lo_i, hi_i, ci in zip(y, lo, hi, cols):
        if np.isfinite(lo_i):
            ax.plot([lo_i, hi_i], [yi, yi], color=ci, lw=1.5, solid_capstyle="butt", zorder=2)
    filled = m["wilcoxon_p_vs_ref"].values < 0.05
    ax.scatter(d[filled], y[filled], color=np.array(cols)[filled], s=54, zorder=3)
    ax.scatter(d[~filled], y[~filled], facecolors="white", edgecolors=np.array(cols)[~filled],
               s=54, zorder=3, linewidths=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(m["key"], fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(-0.8, len(m) - 0.2)
    ax.set_xlabel("Δ Q1+Q2 vs Base — paired on persona, n = 96, 95% CI on the paired difference",
                  fontsize=11)
    ax.set_title("Exp2 — every PTO model state against Base\n"
                 "filled = Wilcoxon p < .05, hollow = not", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT2, "exp2_vs_base_forest.png"))
    plt.close(fig)


def exp2_instrument_grid(df):
    metrics = ["Q1_Mean", "Q2_Mean", "Q1Q2_Mean", "WAI_TotalMean", "CSQ8_Mean", "MI_Mean",
               "MITI_GlobalMean"]
    nice = {"Q1_Mean": "Q1", "Q2_Mean": "Q2", "Q1Q2_Mean": "Q1+Q2",
            "WAI_TotalMean": "WAI-SR", "CSQ8_Mean": "CSQ-8", "MI_Mean": "MI-SAT",
            "MITI_GlobalMean": "MITI global"}
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.0), sharex=True)
    for ax, metric in zip(axes.ravel(), metrics):
        m = _e2(df, metric)
        base = m[m.model_state == "Base"]["mean"].iloc[0]
        ax.axhline(base, ls=":", lw=1.4, color="#888888")
        for orc in ["Q1Q2", "WAI", "CSQ8"]:
            for K, ls, mk in [(0.0, "-", "o"), (5.0, "--", "s")]:
                g = m[(m.train_oracle == orc) & (m.K == K)].sort_values("iteration")
                if g.empty:
                    continue
                ax.plot(g["iteration"], g["mean"], ls=ls, marker=mk, color=ORACLE_C[orc],
                        lw=1.5, ms=4.5, alpha=0.95)
        ax.set_title(nice[metric], fontsize=10.5)
        ax.set_xticks(range(1, 11, 2))
    axes[1, 3].axis("off")
    h = [plt.Line2D([], [], color=ORACLE_C[o], lw=2, label="trained on " + o)
         for o in ["Q1Q2", "WAI", "CSQ8"]]
    h += [plt.Line2D([], [], color="#555555", ls="-", marker="o", label="K=0"),
          plt.Line2D([], [], color="#555555", ls="--", marker="s", label="K=5"),
          plt.Line2D([], [], color="#888888", ls=":", label="Base")]
    axes[1, 3].legend(handles=h, loc="center", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("PTO iteration")
    for ax in axes[:, 0]:
        ax.set_ylabel("score, mean")
    fig.suptitle("Exp2 — all six instruments read on every arm   "
                 "(training reward varies by colour; n = 96 per point)",
                 y=1.01, fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(OUT2, "exp2_instrument_grid.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT1, exist_ok=True)
    os.makedirs(OUT2, exist_ok=True)

    print("Exp1: loading raw conversation + score CSVs ...")
    df1 = _load_exp1()
    print("  %d rows, %d models, %d personas"
          % (len(df1), df1.groupby(["arm", "iteration"]).ngroups, df1["persona"].nunique()))
    exp1_trajectory(df1)
    exp1_paired_delta(df1)
    exp1_distributions(df1)
    exp1_sd_and_length(df1)
    exp1_vs_base(df1)

    print("Exp2: loading _exp2_summary.csv ...")
    df2 = pd.read_csv(EXP2_CSV)
    assert not df2["model_state"].astype(str).str.match(r"^GRPOI?_").any(), \
        "void GRPO V1 states present in _exp2_summary.csv — re-strip before rendering"
    print("  %d model states, %d metrics" % (df2["model_state"].nunique(), df2["metric"].nunique()))
    exp2_trajectory(df2)
    exp2_own_instrument(df2)
    exp2_vs_base_forest(df2)
    exp2_instrument_grid(df2)

    for d in (OUT1, OUT2):
        print("%s:" % os.path.relpath(d, REPO))
        for f in sorted(os.listdir(d)):
            print("   ", f)


if __name__ == "__main__":
    main()
