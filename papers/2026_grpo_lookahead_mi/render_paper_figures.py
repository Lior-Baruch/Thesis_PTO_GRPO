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
    d = pd.read_excel(VALIDITY_XLSX, sheet_name="judge_saturation_grpo_data")
    a = d[(d.panel == "a") & (d.quantity == "cross_judge_pearson_r")]
    med = float(d[(d.panel == "a") & (d.quantity == "cross_judge_pearson_r_median_over_grpo_states")].value.iloc[0])
    b = d[(d.panel == "b") & (d.quantity == "sd_of_per_conversation_score")]
    n_states = int(a.shape[0])

    # Two panels side by side: a full-width figure* costs the page less than a tall column figure.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 1.95))
    # (a) agreement per state
    for arm, s in a.groupby("arm"):
        s = s.sort_values("iteration")
        hot = arm == "GRPO_LA5"
        axA.plot(s.iteration, s.value, color=COL[arm], label=LAB[arm], ms=3.8 if hot else 3.0,
                 lw=1.9 if hot else 1.3, zorder=3 if hot else 2, **STY[arm])
    axA.axhline(med, ls=":", lw=1.0, color="#444444", zorder=1)
    axA.text(0.02, 0.97, f"dotted: median over the {n_states} model states", transform=axA.transAxes,
             fontsize=6.5, color="#444444", va="top", ha="left")
    la5 = a[a.arm == "GRPO_LA5"].sort_values("iteration")
    for it in (9, 10):
        r = float(la5[la5.iteration == it].value.iloc[0])
        axA.annotate(f"{r:.3f}", (it, r), textcoords="offset points", xytext=(0, -11), ha="center",
                     fontsize=7, fontweight="bold", color=COL["GRPO_LA5"])
    axA.set_xticks(range(0, 11))
    axA.set_ylim(0.4, 1.0)
    axA.set_xlabel("iteration (0 = base policy)")
    axA.set_ylabel("per-conversation $r$ on Q1\n(held-out vs training oracle)")
    axA.set_title("(a) cross-grader agreement per model state", loc="left", fontweight="bold")
    axA.legend(frameon=False, loc="lower left")

    # (b) the SD mechanism, K=5 arm, both graders on the same conversations. The judge column
    # holds display names ("gpt-4o-mini", "Claude Haiku 4.5"); match by substring.
    def _is_primary(judge: str) -> bool:
        return "gpt" in judge.lower()

    stats = {}
    series = {}
    for judge, s in b.groupby("judge"):
        s = s.sort_values("iteration")
        rho, p = spearmanr(s.iteration, s.value)
        v0, vN = float(s.value.iloc[0]), float(s.value.iloc[-1])
        stats[judge] = (rho, p, vN ** 2 / v0 ** 2)
        series[judge] = (s, v0, vN)
    # draw the primary first so it sits under the held-out line where they overlap
    for judge in sorted(series, key=lambda j: not _is_primary(j)):
        s, v0, vN = series[judge]
        rho, p, _ = stats[judge]
        prim = _is_primary(judge)
        col = "#d55e00" if prim else "#0072b2"
        who = "training oracle" if prim else "held-out judge"
        axB.plot(s.iteration, s.value, marker="o", ms=3.0, lw=1.5, color=col,
                 label=f"{who}: $\\rho={rho:+.2f}$, $p={p:.3f}$")
        for it, v in ((int(s.iteration.iloc[0]), v0), (int(s.iteration.iloc[-1]), vN)):
            axB.annotate(f"{v:.3f}", (it, v), textcoords="offset points",
                         xytext=(0, 5 if prim else -10), ha="center", fontsize=6.5,
                         fontweight="bold", color=col)
    axB.set_xticks(range(0, 11))
    axB.set_ylim(0.6, 1.65)   # head-room for the legend above the data (max 1.336)
    axB.set_xlabel("iteration (0 = base policy)")
    axB.set_ylabel("SD of per-conversation Q1\n($K{=}5$ arm, each grader's own units)")
    axB.set_title("(b) one ruler saturates, the other does not", loc="left", fontweight="bold")
    axB.legend(frameon=False, loc="upper center", ncol=1, fontsize=6.5,
               title="Spearman trend of SD over iterations", title_fontsize=6.5)
    fig.tight_layout(w_pad=1.6)
    out = DEST / "judge_saturation_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saturation stats (recomputed from the table, must match the text):",
          {k: tuple(round(x, 3) for x in v) for k, v in stats.items()})
    return out


def main() -> int:
    DEST.mkdir(exist_ok=True)
    for f in (overpraise, saturation):
        print("wrote", f())
    return 0


if __name__ == "__main__":
    sys.exit(main())
