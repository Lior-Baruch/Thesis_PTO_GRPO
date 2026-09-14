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
        axA.plot(s.iteration, s.value, color=COL[arm], label=LAB[arm], ms=3.8 if hot else 3.0,
                 lw=1.9 if hot else 1.3, zorder=3 if hot else 2, **STY[arm])
    axA.axhline(med, ls=":", lw=1.0, color="#444444", zorder=1)
    axA.text(0.98, 0.97, f"dotted: median over\nthe {n_states} model states", transform=axA.transAxes,
             fontsize=6.0, color="#444444", va="top", ha="right")
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
    axA.legend(frameon=False, loc="lower left", fontsize=6.0)

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
        if arm == "GRPO_LA5":
            # primary: start label above, end label below; held-out: the reverse, so the two
            # end labels at iteration 10 (0.70 vs 0.91) never collide.
            for it, v, dy in ((int(s.iteration.iloc[0]), v0, 5 if prim else -10),
                              (int(s.iteration.iloc[-1]), vN, -10 if prim else 5)):
                axB.annotate(f"{v:.2f}", (it, v), textcoords="offset points", xytext=(0, dy),
                             ha="center", fontsize=6.2, fontweight="bold", color=JCOL[prim])
            if prim:
                vC = float(s["share_ge45"].iloc[-1])
                axC.annotate(f"{vC:.0%}", (10, vC), textcoords="offset points", xytext=(0, 5),
                             ha="center", fontsize=6.2, fontweight="bold", color=JCOL[prim])
    r5 = {prim: stats[(j, "GRPO_LA5")] for j in sd.judge.unique() for prim in [_is_primary(j)]}
    axB.text(0.03, 0.02,
             "Spearman trend of SD, $K{=}5$:\n"
             f"training oracle $\\rho={r5[True][0]:+.2f}$, $p={r5[True][1]:.3f}$\n"
             f"held-out judge $\\rho={r5[False][0]:+.2f}$, $p={r5[False][1]:.3f}$",
             transform=axB.transAxes, fontsize=5.6, va="bottom", ha="left", color="#333333")
    hb = float(sd[~sd.judge.map(_is_primary)].share_ge45.max())
    held = "0%" if hb == 0 else f"at most {hb:.0%}"
    axC.text(0.5, 0.10, f"held-out judge: {held} at\nevery iteration, both arms",
             transform=axC.transAxes, fontsize=5.8, va="bottom", ha="center", color=JCOL[False])
    for ax in (axB, axC):
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("iteration (0 = base policy)")
    axB.set_ylim(0.36, 1.5)   # room below the data for the trend-test text
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


def main() -> int:
    DEST.mkdir(exist_ok=True)
    for f in (overpraise, saturation):
        print("wrote", f())
    return 0


if __name__ == "__main__":
    sys.exit(main())
