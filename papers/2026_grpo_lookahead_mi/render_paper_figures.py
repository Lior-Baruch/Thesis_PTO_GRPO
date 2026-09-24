"""Draw the paper's four data figures at page proportions, from the EDA's tracked tables.

The EDA's own renders were designed for a notebook, not a two-column page: scaled to the ACL text
width their tick labels fall below 4 pt. This script redraws exactly the same numbers at paper
proportions. **Nothing here computes a number**: every plotted point is read from a tracked table
the EDA rendered beside the original figure (the ``.xlsx`` workbooks under
``Exp3_PTO_GRPO/eda/results/``), so the figures remain EDA-owned in the sense that matters --
re-render the EDA, re-run this, and the picture moves with the table. ``NUMBERS.md`` cites those
tables for every value the captions quote.

    & ..\\..\\.venv\\Scripts\\python.exe render_paper_figures.py

Writes, under ``figures/``: ``levels_grid_grpo_<judge>`` (Fig. 2 and its held-out twin, on the
shared Base since 2026-09-24; ``k_headline_q1q2_grpo`` before that), ``overpraise_judgefree_grpo``
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


def figsize(name: str, aspect: float, default_frac: float = 1.0) -> tuple[float, float]:
    """On-page size of ``name`` at its included width, keeping height/width = ``aspect``.
    ``default_frac`` applies while no section includes the figure yet."""
    w = TEXTWIDTH_IN * FRACS.get(name, default_frac)
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
    """Figure 3 (single column since 2026-09-17): the judge-free lexical over-praise marker by
    iteration, from ``behaviour.xlsx`` sheet ``overpraise_judgefree_data``. The two oracle-coded
    panels it used to carry are superseded by the per-utterance code mix of Figure 4, which shows
    the same drift on both graders with the coder's own praise code."""
    op = pd.read_excel(BEHAVIOUR_XLSX, sheet_name="overpraise_judgefree_data")
    op = op[op.arm.isin(COL)].sort_values(["arm", "iteration"])
    fig, ax = plt.subplots(figsize=figsize("overpraise_judgefree_grpo.png", 0.62))
    for arm, a in op.groupby("arm"):
        ax.plot(a.iteration, a["lex_overpraise_marker_rate"], color=COL[arm], label=LAB[arm],
                ms=3.5, lw=1.4, **STY[arm])
    ax.set_ylabel("share of therapist turns with a marker")
    ax.set_xlabel("iteration")
    ax.set_xticks(range(0, 11, 2))
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="upper left", fontsize=6.0, handlelength=1.5,
              borderaxespad=0.2, labelspacing=0.2, handletextpad=0.4)
    fig.tight_layout()
    out = DEST / "overpraise_judgefree_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- the utterance-level process figures (2026-09-17) -------------------------------------------
PROCESS_XLSX = RESULTS / "lookahead" / "process" / "tables" / "process.xlsx"
TEXT_XLSX = RESULTS / "lookahead" / "text" / "tables" / "text.xlsx"
CODES = ["OQ", "CQ", "SR", "CR", "AF", "PRA", "GI", "PERS", "SEEK", "CONF", "OTH"]
CODE_LABEL = {"OQ": "open question", "CQ": "closed question", "SR": "simple reflection",
              "CR": "complex reflection", "AF": "affirmation", "PRA": "non-specific praise",
              "GI": "giving information", "PERS": "persuasion", "SEEK": "seeking collaboration",
              "CONF": "confront / direct", "OTH": "other"}
# Reflections blue, questions teal, affirmation green, praise vermilion (the turn-level hack),
# information grey, persuasion purple, the rest light.
CODE_COL = {"OQ": "#1b9e77", "CQ": "#a6dbc9", "SR": "#9ecae1", "CR": "#08519c", "AF": "#33a02c",
            "PRA": "#d55e00", "GI": "#bdbdbd", "PERS": "#984ea3", "SEEK": "#f0e442",
            "CONF": "#7f7f7f", "OTH": "#e5e5e5"}
YIELD_CODES = ["CR", "SR", "AF", "GI", "PRA", "PERS"]
BINS = ["1-2", "3-5", "6-9", "10+"]
JUDGE_TITLE = {PRIMARY: "training oracle", HELDOUT: "held-out judge"}


def _process_panels(judge: str, name: str) -> Path:
    """Four panels from ``process.xlsx``: (a, b) each arm's code mix by iteration
    (``process_levels_<judge>``, stacked ``th_<CODE>_rate``), (c) P(change talk | therapist code)
    at iteration 10 with Wilson intervals (``yield_<judge>``; codes with fewer than 20 turns in an
    arm are left blank), (d) change-talk share by patient turn bin at the endpoint against the
    pooled base (``ct_trajectory_<judge>``)."""
    lv = pd.read_excel(PROCESS_XLSX, sheet_name=f"process_levels_{judge}")
    lv = lv[lv.arm.isin(COL)]
    yd = pd.read_excel(PROCESS_XLSX, sheet_name=f"yield_{judge}")
    yd = yd[yd.arm.isin(COL) & (yd.iteration == 10)]
    tr = pd.read_excel(PROCESS_XLSX, sheet_name=f"ct_trajectory_{judge}")
    tr = tr[tr.arm.isin(COL)]
    fig, axes = plt.subplots(1, 4, figsize=figsize(name, 0.23),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.05, 0.95]})
    for ax, arm, title in zip(axes[:2], ("GRPO_LA0", "GRPO_LA5"),
                              ("(a) $K{=}0$: code mix", "(b) $K{=}5$: code mix")):
        g = lv[lv.arm == arm].sort_values("iteration")
        ax.stackplot(g.iteration, *[g[f"th_{c}_rate"].fillna(0).to_numpy() for c in CODES],
                     labels=[CODE_LABEL[c] for c in CODES], colors=[CODE_COL[c] for c in CODES],
                     lw=0.25, edgecolor="white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("iteration")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(False)
    axes[0].set_ylabel("share of therapist turns")
    # (c) yield of each therapist behaviour at the endpoint
    ax = axes[2]
    x = np.arange(len(YIELD_CODES))
    w = 0.38
    for i, arm in enumerate(("GRPO_LA0", "GRPO_LA5")):
        d = yd[yd.arm == arm].set_index("th_code").reindex(YIELD_CODES)
        ok = d["n"].fillna(0) >= 20
        v = d["p_ct"].where(ok)
        err = np.vstack([np.clip(v - d["p_ct_lo"], 0, None).fillna(0),
                         np.clip(d["p_ct_hi"] - v, 0, None).fillna(0)])
        ax.bar(x + (i - 0.5) * w, v, w, color=COL[arm], label=LAB[arm], yerr=err,
               error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels(YIELD_CODES, fontsize=5.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(change talk next)")
    ax.set_xlabel("therapist code, iteration 10")
    ax.set_title("(c) yield of each code", loc="left", fontweight="bold")
    # (d) within-session change talk
    ax = axes[3]
    xb = np.arange(len(BINS))
    base = tr[tr.iteration == 0].groupby("bin")["ct_prop"].mean().reindex(BINS)
    ax.plot(xb, base.values, color="#555555", ls=":", lw=1.3, marker="d", ms=3.2, label="base")
    for arm in ("GRPO_LA0", "GRPO_LA5"):
        g = tr[(tr.arm == arm) & (tr.iteration == 10)].set_index("bin")["ct_prop"].reindex(BINS)
        ax.plot(xb, g.values, color=COL[arm], ms=3.4, lw=1.5, label=LAB[arm], **STY[arm])
    ax.set_xticks(xb)
    ax.set_xticklabels(BINS)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("patient turn in the session")
    ax.set_ylabel("change-talk share")
    ax.set_title("(d) change talk, iter. 10", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=5.6, handlelength=1.2, borderaxespad=0.2)
    # One shared legend for the code colours, above panels (a)-(c): three rows, 11 entries.
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", bbox_to_anchor=(0.40, 0.985), ncol=6, frameon=False,
               fontsize=5.6, handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(w_pad=1.3)
    out = DEST / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def process() -> Path:
    """Figure 4 (body): the utterance-level process picture under the training oracle."""
    return _process_panels(PRIMARY, "process_grpo.png")


def process_heldout() -> Path:
    """Appendix twin of Figure 4 under the held-out judge."""
    return _process_panels(HELDOUT, "process_grpo_heldout.png")


def responsiveness() -> Path:
    """Appendix: what the policy does after the patient's change talk and after sustain talk, by
    iteration, under each grader (``process_levels_<judge>``: ``refl_after_ct``, ``pra_after_st``,
    the per-conversation means)."""
    panels = [(PRIMARY, "refl_after_ct", "(a) reflects CT, oracle"),
              (HELDOUT, "refl_after_ct", "(b) reflects CT, held out"),
              (PRIMARY, "pra_after_st", "(c) praises ST, oracle"),
              (HELDOUT, "pra_after_st", "(d) praises ST, held out")]
    fig, axes = plt.subplots(1, 4, figsize=figsize("responsiveness_grpo.png", 0.26))
    for ax, (judge, col, title) in zip(axes, panels):
        lv = pd.read_excel(PROCESS_XLSX, sheet_name=f"process_levels_{judge}")
        for arm in ("GRPO_LA0", "GRPO_LA5"):
            g = lv[lv.arm == arm].sort_values("iteration")
            ax.fill_between(g.iteration, g[col] - g[f"{col}_se"], g[col] + g[f"{col}_se"],
                            color=COL[arm], alpha=0.18, lw=0)
            ax.plot(g.iteration, g[col], color=COL[arm], label=LAB[arm], ms=3.2, lw=1.4, **STY[arm])
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(0, 0.8)
        ax.set_xlabel("iteration")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=6.6)
    axes[0].set_ylabel("P(reflect | patient CT)")
    axes[2].set_ylabel("P(praise | patient ST)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.3)
    out = DEST / "responsiveness_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def textspace() -> Path:
    """Appendix: the two policies in sentence-embedding space (``text.xlsx``): (a) the cosine
    between their displacements from the base centroid (``drift_cosines``), (b) the
    between-persona share of embedding variance with its bootstrap band and (c) the template
    similarity across personas at matched turn (``diversity_by_state``)."""
    cos = pd.read_excel(TEXT_XLSX, sheet_name="drift_cosines").sort_values("iteration")
    dv = pd.read_excel(TEXT_XLSX, sheet_name="diversity_by_state")
    dv = dv[dv.arm.isin(COL)]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=figsize("text_grpo.png", 0.29))
    a.axhline(0, color="#444444", lw=0.8)
    a.plot(cos.iteration, cos.cos_K0_K5_GRPO, color="#333333", marker="o", ms=3.4, lw=1.5)
    a.set_ylim(-0.2, 1.0)
    a.set_xticks(range(1, 11))
    a.set_xlabel("iteration")
    a.set_ylabel("cosine of the arms' displacements")
    a.set_title("(a) same direction?", loc="left", fontweight="bold")
    for ax, col, ylab, title in ((b, "persona_var_share", "between-persona variance share",
                                  "(b) tailoring to the patient"),
                                 (c, "template_sim", "mean cosine at matched turn",
                                  "(c) convergence on a template")):
        for arm in ("GRPO_LA0", "GRPO_LA5"):
            g = dv[dv.arm == arm].sort_values("iteration")
            if f"{col}_lo" in g.columns:
                ax.fill_between(g.iteration, g[f"{col}_lo"], g[f"{col}_hi"], color=COL[arm], alpha=0.18, lw=0)
            ax.plot(g.iteration, g[col], color=COL[arm], label=LAB[arm], ms=3.2, lw=1.4, **STY[arm])
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
    b.set_ylim(0.1, 0.5)
    c.set_ylim(0.2, 0.65)
    b.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.4)
    out = DEST / "text_grpo.png"
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
    fig, axes = plt.subplots(1, 2, figsize=figsize("k_headline_q1q2_grpo.png", 0.31))
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


SHARED_BASE_XLSX = RESULTS / "lookahead" / "shared_base" / "tables" / "shared_base.xlsx"
GRID_METRICS = [("Q1Q2", "Q1+Q2 (reward)"), ("Q1", "Q1"), ("Q2", "Q2"), ("WAI-SR", "WAI-SR"),
                ("CSQ-8", "CSQ-8"), ("MI-SAT", "MI-SAT"), ("MITI", "MITI"), ("PCT", "PCT"),
                ("MICI", "MICI (lower = better)")]


def _levels_grid(judge: str) -> Path:
    """Every instrument by iteration for one grader, on ONE shared Base (2026-09-24: Lior's single
    Base; the all-instrument grid replaces the Q1+Q2-only headline in section 4). Two rows of five
    slots (nine instruments + the key), mean +/- SE bands; iteration 0 is the Base both runs start
    from (the two base draws pooled, dotted grey line); a star over every iteration whose
    persona-paired K contrast clears Holm across iterations 1..10. Reads
    ``shared_base.xlsx::levels_long`` and ``::k_contrast``."""
    lv = pd.read_excel(SHARED_BASE_XLSX, sheet_name="levels_long")
    kc = pd.read_excel(SHARED_BASE_XLSX, sheet_name="k_contrast")
    lv, kc = lv[lv.judge == judge], kc[kc.judge == judge]
    name = f"levels_grid_grpo_{judge}.png"
    fig, axes = plt.subplots(2, 5, figsize=figsize(name, 0.40, default_frac=0.94))
    flat = axes.ravel()
    for ax, (m, title) in zip(flat, GRID_METRICS):
        d = lv[lv.metric == m]
        for arm in ("GRPO_LA0", "GRPO_LA5"):
            s = d[d.arm == arm].sort_values("iteration")
            ax.fill_between(s.iteration, s["mean"] - s.se, s["mean"] + s.se, color=COL[arm], alpha=0.18, lw=0)
            ax.plot(s.iteration, s["mean"], color=COL[arm], ms=2.4, lw=1.1, **STY[arm])
        base = float(d[d.iteration == 0]["mean"].iloc[0])
        ax.axhline(base, color="#555555", ls=":", lw=0.8)
        lo = float((d["mean"] - d.se).min())
        hi = float((d["mean"] + d.se).max())
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 0.24 * (hi - lo))
        star_y = hi + 0.12 * (hi - lo)
        for it in kc[(kc.metric == m) & (kc.p_holm < 0.05)].iteration:
            ax.text(int(it), star_y, "*", ha="center", va="center", fontsize=7, color="#222222")
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlim(-0.5, 10.5)
        ax.set_title(title, fontsize=6.6, loc="left", fontweight="bold")
        ax.tick_params(labelsize=5.8, pad=1.5)
    for ax in axes[1]:
        ax.set_xlabel("iteration (0 = Base)", fontsize=6.2, labelpad=1.5)
    key = flat[-1]
    key.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL["GRPO_LA0"], ms=3, lw=1.2, label="$K{=}0$", **STY["GRPO_LA0"]),
               Line2D([], [], color=COL["GRPO_LA5"], ms=3, lw=1.2, label="$K{=}5$", **STY["GRPO_LA5"]),
               Line2D([], [], color="#555555", ls=":", lw=0.9, label="Base"),
               Line2D([], [], color="none", marker="$*$", ms=5, markerfacecolor="#222222",
                      markeredgecolor="#222222", label="significant")]
    key.legend(handles=handles, loc="center", fontsize=6.2, frameon=False, handlelength=1.6,
               labelspacing=0.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    out = DEST / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def levels_grid_primary() -> Path:
    return _levels_grid(PRIMARY)


def levels_grid_heldout() -> Path:
    return _levels_grid(HELDOUT)


def faithfulness() -> Path:
    """Appendix B.1 (added 2026-09-22 on Doron's "ref specific subsection and related figure"):
    the faithfulness of the training reward by prefix length, GRPO arms, iterations 1-10 pooled,
    one panel per grader -- from ``mechanism.xlsx`` sheet ``faithfulness_curve_long`` (the rows
    the EDA's four-arm ``faithfulness.png`` draws). The ``all`` (pooled-over-length) rows are not
    drawn. Prints the 12- and 50-utterance values, which must match the B.1 text."""
    d = pd.read_excel(MECHANISM_XLSX, sheet_name="faithfulness_curve_long")
    # n_turns mixes exact lengths with the pooled rows ("all") and the binned ones ("12-20");
    # only the exact lengths are a curve.
    d["n_turns"] = pd.to_numeric(d.n_turns, errors="coerce")
    d = d[d.arm.isin(COL) & d.n_turns.notna()].copy()
    d["n_turns"] = d.n_turns.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=figsize("faithfulness_grpo.png", 0.30), sharey=True)
    panels = ((PRIMARY, "(a) training oracle"), (HELDOUT, "(b) held-out judge"))
    for ax, (judge, title) in zip(axes, panels):
        for arm, g in d[d.judge == judge].groupby("arm"):
            g = g.sort_values("n_turns")
            ax.fill_between(g.n_turns, g.ci_lo, g.ci_hi, color=COL[arm], alpha=0.18, lw=0)
            ax.plot(g.n_turns, g.agreement, color=COL[arm], label=LAB[arm], ms=3.0, lw=1.4,
                    **STY[arm])
        ax.set_xticks(range(10, 51, 10))
        ax.set_xlabel("prefix length (utterances)")
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylim(0.70, 0.96)
    axes[0].set_ylabel("agreement with the\nfull-session ranking")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.0)
    fig.tight_layout(w_pad=1.6)
    out = DEST / "faithfulness_grpo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    at = d[d.n_turns.isin([12, 50])].pivot_table(index=["judge", "arm"], columns="n_turns",
                                                  values="agreement")
    print("faithfulness at 12 / 50 utterances (must match Appendix B.1):")
    print(at.round(3).to_string())
    return out


def main(argv: list[str] | None = None) -> int:
    """Draw every figure, or only the functions named on the command line."""
    DEST.mkdir(exist_ok=True)
    # saturation() is not in the list: its figure left the paper on 2026-09-16 (sec 7's text
    # carries every number it showed). Call it by hand to re-check those numbers.
    # headline() drew the Q1+Q2-only Figure 2 until 2026-09-24; the all-instrument grid on the
    # shared Base (levels_grid_primary) replaced it. Kept, not called, like saturation().
    every = (levels_grid_primary, levels_grid_heldout, overpraise, process, process_heldout,
             responsiveness, textspace, tail_audit, forest, faithfulness)
    names = argv if argv else [f.__name__ for f in every]
    by_name = {f.__name__: f for f in every}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown figure function(s): {unknown}; choose from {sorted(by_name)}")
    for n in names:
        print("wrote", by_name[n]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
