"""Draw the paper's four data figures at page proportions, from the EDA's tracked tables.

The EDA's own renders were designed for a notebook, not a two-column page: scaled to the ACL text
width their tick labels fall below 4 pt. This script redraws exactly the same numbers at paper
proportions. **Nothing here computes a number**: every plotted point is read from a tracked table
the EDA rendered beside the original figure (the ``.xlsx`` workbooks under
``Exp3_PTO_GRPO/eda/results/``), so the figures remain EDA-owned in the sense that matters --
re-render the EDA, re-run this, and the picture moves with the table. ``NUMBERS.md`` cites those
tables for every value the captions quote.

    & ..\\..\\.venv\\Scripts\\python.exe render_paper_figures.py

Writes, under ``figures/``: ``k_headline_q1q2_grpo`` (Fig. 2), ``overpraise_judgefree_grpo``
(Fig. 3), ``k_channel_forest_grpo_gpt-4o-mini`` (Fig. 6) and ``tail_audit_grpo`` (Fig. 7) -- the
same destination names ``sync_figures.py`` used to copy, so the .tex is unchanged;
``sync_figures.py`` no longer lists them. ``saturation()`` (the former Fig. 4, dropped 2026-09-16)
is kept for the Spearman / variance-ratio printout that checks section 7's numbers; ``main()`` does
not call it.

SIZING (2026-09-16): each figure is drawn at the exact width ``sections/*.tex`` includes it at,
so the point sizes in this file are true page point sizes -- see ``width_fracs`` / ``figsize``.
A ``figure*`` pays a fixed text height (title, tick row, x-label) whatever its width, so the
page-space lever is the ASPECT passed to ``figsize``, not the ``\\textwidth`` fraction: narrowing a
figure only shrinks its type. Re-run after any width change; then rebuild and confirm the body
still ends on page 8.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "Exp3_PTO_GRPO" / "eda" / "results"
BEHAVIOUR_XLSX = RESULTS / "lookahead" / "behaviour" / "tables" / "behaviour.xlsx"
VALIDITY_XLSX = RESULTS / "measurement" / "validity" / "tables" / "validity.xlsx"
REPLICATION_XLSX = RESULTS / "lookahead" / "replication" / "tables" / "replication.xlsx"
MECHANISM_XLSX = RESULTS / "lookahead" / "mechanism" / "tables" / "mechanism.xlsx"
REWARD_XLSX = RESULTS / "lookahead" / "reward" / "tables" / "reward.xlsx"
DEST = HERE / "figures"
SECTIONS = HERE / "sections"

# ACL \textwidth, read off main.log ("* \textwidth=455.24411pt", TeX points).
TEXTWIDTH_IN = 455.24411 / 72.27

# --- width-aware sizing (2026-09-16) ------------------------------------------------------------
# Each figure is drawn at its EXACT on-page size, so \includegraphics scales it by 1.0 and every
# point size in this file is a TRUE PAGE POINT SIZE. Before this, every figure was drawn 6.3 in
# wide whatever width it was included at, so a figure at 0.64\textwidth printed its 7.5 pt labels
# at 4.8 pt. The fractions are parsed from the .tex rather than hardcoded, so narrowing a figure
# in the source re-sizes its render on the next run instead of silently shrinking its type.
_INCLUDE_RE = re.compile(
    r"\\includegraphics\[width=([0-9.]+)\\textwidth\]\{figures/([^}]+)\}")


def width_fracs() -> dict[str, float]:
    """name -> the \\textwidth fraction sections/*.tex includes it at (1.0 if never included)."""
    out: dict[str, float] = {}
    for tex in sorted(SECTIONS.glob("*.tex")):
        for frac, name in _INCLUDE_RE.findall(tex.read_text(encoding="utf-8")):
            out[name] = float(frac)
    return out


FRACS = width_fracs()


def figsize(name: str, aspect: float) -> tuple[float, float]:
    """On-page size of ``name`` at its included width, keeping height/width = ``aspect``."""
    w = TEXTWIDTH_IN * FRACS.get(name, 1.0)
    return (w, w * aspect)


# Same two arm colours as the EDA's headline figure (Okabe-Ito vermilion / orange).
COL = {"GRPO_LA0": "#d55e00", "GRPO_LA5": "#e69f00"}
LAB = {"GRPO_LA0": "$K{=}0$ (turn-level)", "GRPO_LA5": "$K{=}5$ (look-ahead)"}
STY = {"GRPO_LA0": dict(marker="o", ls="-"), "GRPO_LA5": dict(marker="s", ls="--")}
PRIMARY = "gpt-4o-mini"
HELDOUT = "claude-haiku-4-5"

# TRUE PAGE POINT SIZES (see above). ACL body text is 11 pt and captions 10 pt; 6-7 pt is the
# floor at which a scaled screenshot of the PDF still reads. Nothing here goes below 5.8.
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 6.8, "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2, "legend.fontsize": 6.2, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def overpraise() -> Path:
    op = pd.read_excel(BEHAVIOUR_XLSX, sheet_name="overpraise_judgefree_data")
    op = op[op.arm.isin(COL)].sort_values(["arm", "iteration"])
    # Panel titles and axis labels are kept SHORT: the type is true 7 pt and each panel is one
    # third of the text width, and what each panel measures is spelled out in the caption.
    panels = [
        ("lex_overpraise_marker_rate", "(a) lexical marker", "share of turns"),
        (f"MICI_OverPraiseRate_{PRIMARY}", "(b) training oracle", "acts per turn"),
        (f"MICI_OverPraiseRate_{HELDOUT}", "(c) held-out judge", "acts per turn"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize("overpraise_judgefree_grpo.png", 0.29))
    for ax, (col, title, ylab) in zip(axes, panels):
        for arm, a in op.groupby("arm"):
            ax.plot(a.iteration, a[col], color=COL[arm], label=LAB[arm], ms=3.5, lw=1.4, **STY[arm])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xlabel("iteration")
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(bottom=0)
    # Legend INSIDE panel (a): all three series rise from ~0, so the upper left is empty, and a
    # legend row above the figure costs ~0.2 in of a page-width float that the body cannot spare.
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.0, handlelength=1.5,
                   borderaxespad=0.2, labelspacing=0.2, handletextpad=0.4)
    fig.tight_layout(w_pad=1.2)
    out = DEST / "overpraise_judgefree_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def saturation() -> Path:
    """Three panels: (a) per-state cross-grader agreement, (b) each grader's Q1 spread along BOTH
    arms, (c) the share of conversations at or above 4.5 on Q1 (the ceiling). Panel (a) reads
    ``validity.xlsx::judge_saturation_grpo_data``; (b) and (c) read
    ``lookahead/replication/tables/replication.xlsx::sd_by_iter`` (metric Q1), which carries the
    K=0 arm too -- the 2026-09-04 figure showed the K=5 arm only, and the K=0 series is what
    exposes the spread as a function of LEVEL (ceiling) rather than of horizon."""
    d = pd.read_excel(VALIDITY_XLSX, sheet_name="judge_saturation_grpo_data")
    a = d[(d.panel == "a") & (d.quantity == "cross_judge_pearson_r")]
    med = float(d[(d.panel == "a") & (d.quantity == "cross_judge_pearson_r_median_over_grpo_states")].value.iloc[0])
    n_states = int(a.shape[0])
    sd = pd.read_excel(REPLICATION_XLSX, sheet_name="sd_by_iter")
    sd = sd[(sd.metric == "Q1") & (sd.arm.isin(COL))]
    sd = sd.drop_duplicates(subset=["judge", "arm", "iteration"]).sort_values(["judge", "arm", "iteration"])

    def _is_primary(judge: str) -> bool:
        return "gpt" in str(judge).lower()

    JCOL = {True: "#d55e00", False: "#0072b2"}        # grader -> colour (primary vermilion, held-out blue)
    ASTY = {"GRPO_LA5": dict(ls="-", lw=1.7, marker="s", ms=3.2, alpha=1.0, zorder=3),
            "GRPO_LA0": dict(ls="--", lw=1.1, marker="o", ms=2.6, alpha=0.75, zorder=2)}
    JWHO = {True: "training oracle", False: "held-out judge"}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=figsize("judge_saturation_grpo.png", 0.24))
    # (a) agreement per state
    for arm, s in a.groupby("arm"):
        s = s.sort_values("iteration")
        hot = arm == "GRPO_LA5"
        axA.plot(s.iteration, s.value, color=COL[arm], ms=3.8 if hot else 3.0,
                 label="$K{=}5$" if hot else "$K{=}0$",
                 lw=1.9 if hot else 1.3, zorder=3 if hot else 2, **STY[arm])
    # The dotted median line is named in the caption, not in the legend: a third entry makes the
    # legend box as wide as this (0.68\textwidth / 3) panel and it then covers the K=5 line.
    axA.axhline(med, ls=":", lw=1.0, color="#444444", zorder=1)
    la5 = a[a.arm == "GRPO_LA5"].sort_values("iteration")
    # Both minima annotated to the RIGHT of their points, on separate rows: the panel is one
    # third of 0.68\textwidth, so a label extending leftward from iteration 9 reaches the legend.
    # The x-limit is opened past 10 to hold them.
    for it, dy in ((9, -8), (10, 7)):
        r = float(la5[la5.iteration == it].value.iloc[0])
        axA.annotate(f"{r:.3f}", (it, r), textcoords="offset points", xytext=(3, dy), ha="left",
                     fontsize=6.2, fontweight="bold", color=COL["GRPO_LA5"])
    axA.set_xlim(-0.7, 13.6)
    axA.set_xticks(range(0, 11, 2))
    axA.set_ylim(0.36, 1.02)
    axA.set_xlabel("iteration")
    axA.set_ylabel("per-conversation $r$ on Q1")
    axA.set_title("(a) agreement", loc="left", fontweight="bold")
    # Lower-left is the one empty region: both lines stay above 0.74 until K=5 dives at 8-10,
    # and the two annotations sit bottom-RIGHT, under that dive.
    axA.legend(frameon=False, loc="lower left", fontsize=5.8, ncol=1, handlelength=1.4,
               borderaxespad=0.15, labelspacing=0.15, handletextpad=0.4)

    # (b) the spread of each grader's Q1 scores along both arms; (c) the ceiling share.
    stats = {}
    handles = {}
    for (judge, arm), s in sd.groupby(["judge", "arm"]):
        s = s.sort_values("iteration")
        prim = _is_primary(judge)
        rho, p = spearmanr(s.iteration, s["sd"])
        v0, vN = float(s["sd"].iloc[0]), float(s["sd"].iloc[-1])
        stats[(judge, arm)] = (rho, p, vN ** 2 / v0 ** 2)
        lab = f"{JWHO[prim]}, $K{{=}}{5 if arm == 'GRPO_LA5' else 0}$"
        (h,) = axB.plot(s.iteration, s["sd"], color=JCOL[prim], label=lab, **ASTY[arm])
        handles[(prim, arm)] = h
        axC.plot(s.iteration, s["share_ge45"], color=JCOL[prim], label=lab, **ASTY[arm])
        if arm == "GRPO_LA5" and prim:
            vC = float(s["share_ge45"].iloc[-1])
            axC.annotate(f"{vC:.0%}", (10, vC), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=6.2, fontweight="bold", color=JCOL[prim])
    # The SD endpoints (1.34 -> 0.70; 0.76 -> 0.91) and the Spearman trends are quoted in the
    # section text and the caption; printed inside panel (b) they collided with the lines.
    hb = float(sd[~sd.judge.map(_is_primary)].share_ge45.max())
    held = "0%" if hb == 0 else f"at most {hb:.0%}"
    axC.text(0.03, 0.97, f"held out: {held},\nboth arms",
             transform=axC.transAxes, fontsize=5.8, va="top", ha="left", color=JCOL[False])
    for ax in (axB, axC):
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("iteration")
    axB.set_ylim(0.5, 1.45)
    # (b)/(c) y-labels kept short enough not to reach the legend row above the axes.
    axB.set_ylabel("SD of Q1 scores")
    axB.set_title("(b) spread", loc="left", fontweight="bold")
    axC.set_ylim(0, 0.72)
    axC.set_ylabel("share of Q1 $\\geq 4.5$")
    axC.set_title("(c) ceiling", loc="left", fontweight="bold")
    order = [(True, "GRPO_LA5"), (True, "GRPO_LA0"), (False, "GRPO_LA5"), (False, "GRPO_LA0")]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
               fontsize=6.0, handlelength=1.9, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.1)
    out = DEST / "judge_saturation_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saturation stats (Spearman rho, p, variance ratio end/start; must match the text):",
          {k: tuple(round(x, 3) for x in v) for k, v in stats.items()})
    return out


def tail_audit() -> Path:
    """Figure 7 (Appendix A): the K=5 rollout audit, from ``mechanism.xlsx`` sheets
    ``tail_audit_by_iter``, ``tail_score_by_realized_turns`` and ``tail_within_group`` (the
    ``GRPO_LA5`` rows). The EDA's own render carried its jargon ("tails", arm codes, "ther. end");
    this one says in words what each panel shows. Nothing is computed here beyond reading."""
    by_iter = pd.read_excel(MECHANISM_XLSX, sheet_name="tail_audit_by_iter")
    by_rt = pd.read_excel(MECHANISM_XLSX, sheet_name="tail_score_by_realized_turns")
    within = pd.read_excel(MECHANISM_XLSX, sheet_name="tail_within_group")
    for d in (by_iter, by_rt, within):
        d["train_iter"] = d["train_iter"].astype(str)
    bi = by_iter[(by_iter.arm == "GRPO_LA5") & (by_iter.train_iter != "pooled")].copy()
    bi["it"] = bi.train_iter.astype(int)
    bi = bi.sort_values("it")
    pooled = by_iter[(by_iter.arm == "GRPO_LA5") & (by_iter.train_iter == "pooled")].iloc[0]
    rt = by_rt[(by_rt.arm == "GRPO_LA5") & (by_rt.train_iter == "pooled")].sort_values("realized_turns")
    wi = within[(within.arm == "GRPO_LA5") & (within.train_iter != "pooled")].copy()
    wi["it"] = wi.train_iter.astype(int)
    wi = wi.sort_values("it")
    col = COL["GRPO_LA5"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=figsize("tail_audit_grpo.png", 0.3333))
    # (a) how often the rollout was cut short
    a.fill_between(bi.it, bi.ended_early_ci_lo, bi.ended_early_ci_hi, color=col, alpha=0.2, lw=0)
    a.plot(bi.it, bi.ended_early_rate, color=col, marker="s", ms=3.2, lw=1.6,
           label="cut short (< 5 turns)")
    a.plot(bi.it, bi.patient_closed_share, color=col, ls=":", lw=1.3,
           label="of which: patient closed")
    a.set_ylim(0, 0.45)
    a.set_xticks(range(1, 11))
    a.set_xlabel("training iteration")
    a.set_ylabel("share of $K{=}5$ rollouts")
    a.set_title("(a) rollouts cut short", loc="left", fontweight="bold")
    a.legend(frameon=False, loc="upper left", fontsize=5.6)
    # (b) what a cut-short rollout cost the candidate. Tick labels: turns actually simulated and
    # who ended the rollout (pat. = the patient closed the session, pol. = the policy's own turn).
    labels = {0: "0\nnone", 1: "1\npat.", 2: "2\npol.", 3: "3\npat.", 4: "4\npol.", 5: "5\nfull"}
    x = rt.realized_turns.astype(int).tolist()
    b.axhline(0, color="#444444", lw=0.8)
    b.errorbar(x, rt.dev_mean, yerr=[rt.dev_mean - rt.dev_lo, rt.dev_hi - rt.dev_mean], fmt="s",
               color=col, ms=3.5, capsize=2, lw=1.2)
    b.set_xticks(x)
    b.set_xticklabels([labels[i] for i in x], fontsize=6.0)
    b.set_ylim(-0.115, 0.025)
    b.set_xlabel("turns simulated, and who ended it")
    b.set_ylabel("candidate reward $-$ group mean\n(Q1+Q2 points)")
    b.set_title("(b) cost of a cut-short rollout", loc="left", fontweight="bold")
    # (c) was the candidate still the group's best?
    c.axhline(0.125, ls=":", color="#444444", lw=1.0, label="chance ($1/8$)")
    c.plot(wi.it, wi.p_chosen_given_full, color=col, marker="o", mfc="white", ms=3.4, lw=1.4,
           label="ran all five turns")
    c.plot(wi.it, wi.p_chosen_given_ee, color=col, marker="s", ms=3.2, lw=1.4, ls="--",
           label="cut short")
    c.set_ylim(0, 0.165)
    c.set_xticks(range(1, 11))
    c.set_xlabel("training iteration")
    c.set_ylabel("share of candidates that\nwere their group's best")
    c.set_title("(c) still the group's best?", loc="left", fontweight="bold")
    c.legend(frameon=False, loc="lower left", fontsize=5.6)
    fig.tight_layout(w_pad=1.6)
    out = DEST / "tail_audit_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("tail audit pooled (must match the caption): ended early",
          round(float(pooled.ended_early_rate), 3), "patient closed",
          round(float(pooled.patient_closed_share), 3), "n", int(pooled.n_candidates))
    return out


# Figure 6: every behaviour channel at iteration 10, in the PAPER's sign (K=5 - K=0), with plain
# labels. (sheet, metric, label, group). Channels that are zero in both arms (confront, warn) are
# omitted; the per-session duplicates of the per-turn rates are omitted for space.
FOREST_ROWS = [
    ("coder", "MICI_Rate", "MI-inconsistent acts per turn", "mici"),
    ("coder", "MICI_OverPraise_rate", "over-praise per turn", "mici"),
    ("coder", "MICI_AdviseNoPermission_rate", "advice without permission per turn", "mici"),
    ("coder", "MICI_Direct_rate", "direct / order per turn", "mici"),
    ("coder", "MICI_Judge_rate", "judge / label per turn", "mici"),
    ("coder", "MICI_Severity", "MI-inconsistency severity (1-5)", "mici"),
    ("coder", "B3_Q_per_turn", "questions per turn", "miti"),
    ("coder", "B4_SR_per_turn", "simple reflections per turn", "miti"),
    ("coder", "B5_CR_per_turn", "complex reflections per turn", "miti"),
    ("coder", "B6_AF_per_turn", "affirmations per turn", "miti"),
    ("coder", "B7_Seek_per_turn", "seeking collaboration per turn", "miti"),
    ("coder", "B1_GI_per_turn", "giving information per turn", "miti"),
    ("coder", "B2_Persuade_per_turn", "persuasion per turn", "miti"),
    ("coder", "%MICO", "share of MI-consistent codes", "miti"),
    ("coder", "RtoQ", "reflection-to-question ratio", "miti"),
    ("coder", "%CR", "share of complex reflections", "miti"),
    ("text", "q_per_turn", "question marks per therapist turn", "shape"),
    ("text", "n_th_turns", "therapist turns per session", "shape"),
    ("text", "conv_len", "utterances per session", "shape"),
    ("text", "mean_turn_len", "characters per therapist turn", "shape"),
]
GROUP_TITLE = {"mici": "MI-inconsistent behaviour (lower is better)",
               "miti": "MITI behaviour codes", "shape": "session shape (from the text)"}
GROUP_COL = {"mici": "#d55e00", "miti": "#0072b2", "shape": "#6e6e6e"}


def forest() -> Path:
    """Figure 6 (Appendix A): the channel forest at iteration 10 from ``behaviour.xlsx`` sheets
    ``k_channels_grpo_gpt-4o-mini`` and ``k_channels_text_grpo``. The sheets store K=0 - K=5; the
    paper reports K=5 - K=0, so every dz is negated here and the axis says so."""
    coder = pd.read_excel(BEHAVIOUR_XLSX, sheet_name="k_channels_grpo_gpt-4o-mini")
    text = pd.read_excel(BEHAVIOUR_XLSX, sheet_name="k_channels_text_grpo")
    src = {"coder": coder[coder.iteration == 10].set_index("metric"),
           "text": text[text.iteration == 10].set_index("metric")}
    rows = []
    for sheet, metric, label, group in FOREST_ROWS:
        r = src[sheet].loc[metric]
        rows.append((label, group, -float(r.dz), float(r.p_holm)))   # sign flipped to K=5 - K=0
    xlim = (-2.8, 2.6)
    fig, ax = plt.subplots(figsize=figsize("k_channel_forest_grpo_gpt-4o-mini.png", 0.5714))
    y = 0
    ys, labels = [], []
    for g in ("mici", "miti", "shape"):
        ax.text(xlim[0] + 0.06, y + 0.15, GROUP_TITLE[g], fontsize=6.4, fontweight="bold",
                color=GROUP_COL[g], ha="left", va="bottom")
        y -= 1
        for label, group, dz, p in rows:
            if group != g:
                continue
            sig = p < 0.05
            ax.barh(y, dz, height=0.72, color=GROUP_COL[g] if sig else "white",
                    edgecolor=GROUP_COL[g], linewidth=1.0)
            ax.annotate(f"{dz:+.2f}{'' if sig else ' (n.s.)'}", (dz, y), textcoords="offset points",
                        xytext=(4 if dz >= 0 else -4, 0), ha="left" if dz >= 0 else "right",
                        va="center", fontsize=5.8, color="#333333")
            ys.append(y)
            labels.append(label)
            y -= 1
        y -= 0.4
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlim(*xlim)
    ax.set_ylim(y + 0.2, 0.9)
    ax.set_xlabel("persona-paired $d_z$, $K{=}5 - K{=}0$   (positive: the look-ahead policy does more of it)")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.01, "hollow bar: did not clear Holm ($p \\geq .05$)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.8, color="#555555")
    fig.tight_layout()
    out = DEST / "k_channel_forest_grpo_gpt-4o-mini.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def headline() -> Path:
    """Figure 2: Q1+Q2 by iteration under each grader, from ``reward.xlsx`` sheet
    ``k_headline_grpo_data`` (the EDA's render is notebook-proportioned and its legend prints at
    ~5 pt at column width). Mean +/- SE bands, each arm's own base dotted, a star over every
    iteration whose persona-paired K=5 vs K=0 contrast clears Holm, endpoint means printed."""
    d = pd.read_excel(REWARD_XLSX, sheet_name="k_headline_grpo_data")
    d = d[d.metric == "Q1Q2"]
    panels = [(PRIMARY, "(a) training oracle (gpt-4o-mini)"), (HELDOUT, "(b) held-out judge (Claude Haiku 4.5)")]
    fig, axes = plt.subplots(1, 2, figsize=figsize("k_headline_q1q2_grpo.png", 0.34))
    for ax, (judge, title) in zip(axes, panels):
        s = d[d.judge == judge].sort_values("iteration")
        for arm, mean, se, base in (("GRPO_LA0", "mean_K0", "se_K0", "base_K0"),
                                    ("GRPO_LA5", "mean_K5", "se_K5", "base_K5")):
            ax.fill_between(s.iteration, s[mean] - s[se], s[mean] + s[se], color=COL[arm], alpha=0.18, lw=0)
            ax.plot(s.iteration, s[mean], color=COL[arm], label=LAB[arm], ms=3.4, lw=1.5, **STY[arm])
            ax.axhline(float(s[base].iloc[0]), color=COL[arm], ls=":", lw=0.9, alpha=0.8)
            v = float(s[mean].iloc[-1])
            ax.annotate(f"{v:.2f}", (10, v), textcoords="offset points", xytext=(4, 0), ha="left",
                        va="center", fontsize=6.5, fontweight="bold", color=COL[arm])
        lo = float(min(s.mean_K0.min(), s.mean_K5.min())) - 0.25
        hi = float(max(s.mean_K0.max(), s.mean_K5.max())) + 0.25
        # Headroom for the star row AND, in panel (a), the legend that now sits inside the axes.
        # The legend takes a fixed FRACTION f of the axes height, so the top margin m has to
        # satisfy m - f(1 + m) > 0.06 to clear the stars; at aspect 0.30 f ~ 0.19, so m > 0.31.
        ax.set_ylim(lo, hi + 0.42 * (hi - lo))
        star_y = hi + 0.06 * (hi - lo)
        for it, sig in zip(s.iteration, s.holm_sig):
            if bool(sig):
                ax.text(it, star_y, "*", ha="center", va="center", fontsize=8, color="#222222")
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlim(-0.4, 11.2)
        ax.set_xlabel("iteration (0 = each arm's own base)")
        ax.set_ylabel("Q1+Q2 (mean $\\pm$ SE)")
        ax.set_title(title, loc="left", fontweight="bold")
    # Two entries only, INSIDE panel (a), stacked: a legend row above the figure costs ~0.2 in of
    # a page-width float the body cannot spare, and a four-entry box is wide enough to reach the
    # star at iteration 4 whatever the top margin. The dotted base line and the star are defined
    # in the caption instead.
    # Bare "K=0" / "K=5" here (the parenthetical glosses reach the iteration-4 star); the arms
    # are the paper's central notation by this point and the caption states the contrast.
    hh, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(hh, ["$K{=}0$", "$K{=}5$"], loc="upper left", fontsize=5.8, ncol=1,
                   frameon=False, handlelength=1.1, handletextpad=0.3, labelspacing=0.2,
                   borderaxespad=0.1)
    fig.tight_layout(w_pad=1.4)
    out = DEST / "k_headline_q1q2_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    DEST.mkdir(exist_ok=True)
    # saturation() is not in the list: its figure left the paper on 2026-09-16 (sec 7's text
    # carries every number it showed). Call it by hand to re-check those numbers.
    for f in (headline, overpraise, tail_audit, forest):
        print("wrote", f())
    return 0


if __name__ == "__main__":
    sys.exit(main())
