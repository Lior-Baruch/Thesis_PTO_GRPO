"""Draw the two BODY figures whose EDA renders are not legible at ACL column/text width.

The EDA renders of ``overpraise_judgefree_grpo`` (3.7:1 aspect, 9-pt titles) and
``judge_saturation_grpo`` (a 7.4 x 7.6 in two-panel figure) were designed for a notebook, not a
two-column page: scaled to the ACL text/column width their tick labels fall below 4 pt. This script
redraws exactly the same numbers at paper proportions. **Nothing here computes a number**: every
plotted point is read from the tracked table the EDA rendered beside the original figure
(``lookahead/behaviour/tables/behaviour.xlsx`` sheet ``overpraise_judgefree_data`` and
``measurement/validity/tables/validity.xlsx`` sheet ``judge_saturation_grpo_data``), so the figures
remain EDA-owned in the sense that matters -- re-render the EDA, re-run this, and the picture moves
with the table. ``NUMBERS.md`` cites those tables for every value the captions quote.

    & ..\\..\\.venv\\Scripts\\python.exe render_paper_figures.py

Writes ``figures/overpraise_judgefree_grpo.png`` and ``figures/judge_saturation_grpo.png`` (the same
destination names ``sync_figures.py`` used to copy, so the .tex is unchanged); ``sync_figures.py`` no
longer lists those two.
"""

from __future__ import annotations

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

# Same two arm colours as the EDA's headline figure (Okabe-Ito vermilion / orange).
COL = {"GRPO_LA0": "#d55e00", "GRPO_LA5": "#e69f00"}
LAB = {"GRPO_LA0": "$K{=}0$ (turn-level reward)", "GRPO_LA5": "$K{=}5$ (look-ahead reward)"}
STY = {"GRPO_LA0": dict(marker="o", ls="-"), "GRPO_LA5": dict(marker="s", ls="--")}
PRIMARY = "gpt-4o-mini"
HELDOUT = "claude-haiku-4-5"

# Sized for placement at ~0.86 of the ACL text width (6.3 in -> ~5.4 in), where these point sizes
# land at 6-7 pt on the page.
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def overpraise() -> Path:
    op = pd.read_excel(BEHAVIOUR_XLSX, sheet_name="overpraise_judgefree_data")
    op = op[op.arm.isin(COL)].sort_values(["arm", "iteration"])
    panels = [
        ("lex_overpraise_marker_rate", "(a) judge-free lexical marker",
         "share of therapist turns\nwith an over-praise marker"),
        (f"MICI_OverPraiseRate_{PRIMARY}", "(b) training oracle",
         "coded over-praise acts\nper therapist turn"),
        (f"MICI_OverPraiseRate_{HELDOUT}", "(c) held-out judge",
         "coded over-praise acts\nper therapist turn"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.8))
    for ax, (col, title, ylab) in zip(axes, panels):
        for arm, a in op.groupby("arm"):
            ax.plot(a.iteration, a[col], color=COL[arm], label=LAB[arm], ms=3.5, lw=1.4, **STY[arm])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xlabel("training iteration")
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6)
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

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(6.3, 2.05))
    # (a) agreement per state
    for arm, s in a.groupby("arm"):
        s = s.sort_values("iteration")
        hot = arm == "GRPO_LA5"
        axA.plot(s.iteration, s.value, color=COL[arm], ms=3.8 if hot else 3.0,
                 label="$K{=}5$ (look-ahead)" if hot else "$K{=}0$ (turn-level)",
                 lw=1.9 if hot else 1.3, zorder=3 if hot else 2, **STY[arm])
    axA.axhline(med, ls=":", lw=1.0, color="#444444", zorder=1,
                label=f"median, {n_states} states")
    la5 = a[a.arm == "GRPO_LA5"].sort_values("iteration")
    for it, dy in ((9, -11), (10, 6)):
        r = float(la5[la5.iteration == it].value.iloc[0])
        axA.annotate(f"{r:.3f}", (it, r), textcoords="offset points", xytext=(0, dy), ha="center",
                     fontsize=6.5, fontweight="bold", color=COL["GRPO_LA5"])
    axA.set_xticks(range(0, 11, 2))
    axA.set_ylim(0.4, 1.0)
    axA.set_xlabel("iteration (0 = base policy)")
    axA.set_ylabel("per-conversation $r$ on Q1\n(held-out vs training oracle)")
    axA.set_title("(a) cross-grader agreement", loc="left", fontweight="bold")
    # Lower-left is the one empty region: both lines stay above 0.74 until K=5 dives at 8-10,
    # and the short labels keep the legend clear of the 0.487 / 0.544 annotations.
    axA.legend(frameon=False, loc="lower left", fontsize=5.6, ncol=1, handlelength=2.0,
               borderaxespad=0.3, labelspacing=0.25)

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
    axC.text(0.03, 0.96, f"held-out judge: {held} at\nevery iteration, both arms",
             transform=axC.transAxes, fontsize=5.8, va="top", ha="left", color=JCOL[False])
    for ax in (axB, axC):
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("iteration (0 = base policy)")
    axB.set_ylim(0.5, 1.45)
    axB.set_ylabel("SD of per-conversation Q1\n(each grader's own units)")
    axB.set_title("(b) spread tracks level", loc="left", fontweight="bold")
    axC.set_ylim(0, 0.72)
    axC.set_ylabel("share of conversations\nscored $\\geq 4.5$ on Q1")
    axC.set_title("(c) the ceiling", loc="left", fontweight="bold")
    order = [(True, "GRPO_LA5"), (True, "GRPO_LA0"), (False, "GRPO_LA5"), (False, "GRPO_LA0")]
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
               fontsize=6.3, handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.3)
    out = DEST / "judge_saturation_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saturation stats (Spearman rho, p, variance ratio end/start; must match the text):",
          {k: tuple(round(x, 3) for x in v) for k, v in stats.items()})
    return out


def tail_audit() -> Path:
    """Figure 8 (Appendix A): the K=5 rollout audit, from ``mechanism.xlsx`` sheets
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

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.3, 2.1))
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


# Figure 7: every behaviour channel at iteration 10, in the PAPER's sign (K=5 - K=0), with plain
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
    """Figure 7 (Appendix A): the channel forest at iteration 10 from ``behaviour.xlsx`` sheets
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
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
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
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.25))
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
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))
        star_y = hi + 0.06 * (hi - lo)
        for it, sig in zip(s.iteration, s.holm_sig):
            if bool(sig):
                ax.text(it, star_y, "*", ha="center", va="center", fontsize=8, color="#222222")
        ax.set_xticks(range(0, 11))
        ax.set_xlim(-0.4, 11.2)
        ax.set_xlabel("iteration (0 = each arm's own base draw)")
        ax.set_ylabel("Q1+Q2, mean $\\pm$ SE over 96 personas")
        ax.set_title(title, loc="left", fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([0], [0], color="#555555", ls=":", lw=0.9))
    l.append("each arm's own base (iteration-0 mean)")
    h.append(plt.Line2D([0], [0], marker="$*$", color="#222222", ls="none", ms=6))
    l.append("$K{=}5$ vs $K{=}0$ clears Holm ($p<.05$)")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, frameon=False,
               fontsize=6.3, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    out = DEST / "k_headline_q1q2_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    DEST.mkdir(exist_ok=True)
    for f in (headline, overpraise, saturation, tail_audit, forest):
        print("wrote", f())
    return 0


if __name__ == "__main__":
    sys.exit(main())
