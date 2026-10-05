"""Figures for the 2026-10-05 check of Doron's win/lose-direction deck.

Reads the tables written by check_doron_directions_2026-10-05.py (no embeddings, no GPU) and writes
four PNGs to meetings/2026-10-05_doron_direction_check/figures/:

  artefacts.png         leaked-marker and degenerate shares among best vs worst candidates
  lose_end_markers.png  marker share among the 10 lowest-projecting pool sentences, per variant
  cosines.png           K0-vs-K5 cosine, reliability and consecutive-iteration stability, raw vs corrected
  features.png          text-feature correlations with the projection on K0 - K5, per arm

Usage: .venv/Scripts/python.exe meetings/build/plot_doron_direction_check_2026-10-05.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "2026-10-05_doron_direction_check")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")

ARMS = ["GRPO_K0", "GRPO_K5"]
LAB = {"GRPO_K0": "K=0", "GRPO_K5": "K=5"}
COL = {"GRPO_K0": "#d55e00", "GRPO_K5": "#e69f00"}  # the paper's arm colours
MARK = {"GRPO_K0": "o", "GRPO_K5": "s"}
BEST, WORST = "#1c5cab", "#86b6ef"  # one blue hue, dark = best (validated as an ordinal pair)
PAIR = "#1c5cab"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGE = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f6c1bd", "#e34948", "#8a1f1e"])
ITERS = list(range(1, 11))
LW, MS = 1.8, 6.5


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8.5,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def line(ax, x, y, color, ls="-", marker="o", hollow=False, label=None):
    ax.plot(x, y, color=color, ls=ls, lw=LW, marker=marker, ms=MS, label=label,
            mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE, mew=1.4 if hollow else 1.2)


def end_label(ax, x, y, text, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8.5, color=INK)


def heading(fig, title, sub, y=1.0):
    """Title + subtitle stacked above figure height y, spaced in points so multi-line subtitles never collide."""
    def at(pts):
        return offset_copy(fig.transFigure, fig=fig, y=pts, units="points")
    fig.text(0.0, y, sub, fontsize=8.5, color=INK2, ha="left", va="bottom", linespacing=1.35, transform=at(6))
    fig.text(0.0, y, title, fontsize=11.5, fontweight="bold", color=INK, ha="left", va="bottom",
             transform=at(6 + 12 * (sub.count("\n") + 1) + 5))


def iter_axis(ax):
    ax.set_xticks(ITERS)
    ax.set_xlim(0.6, 10.4)
    ax.grid(axis="x", visible=False)


def fig_artefacts():
    a = pd.read_csv(os.path.join(TAB, "artefact_rates.csv"))
    a = a[a.iteration != "all"].assign(iteration=lambda d: d.iteration.astype(int))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True, sharey="row")
    rows = [("leak", "Leaked chat marker (<|im_…)"), ("degenerate", "Degenerate text")]
    for r, (key, name) in enumerate(rows):
        for c, arm in enumerate(ARMS):
            ax, g = axes[r, c], a[a.arm == arm]
            line(ax, g.iteration, 100 * g[f"{key}_among_best"], BEST, marker="o", label="best candidate(s)")
            line(ax, g.iteration, 100 * g[f"{key}_among_worst"], WORST, ls="--", marker="s", label="worst candidate(s)")
            ax.set_title(f"{name} · {LAB[arm]}")
            iter_axis(ax)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        axes[r, 0].set_ylabel("share of candidates")
    for ax in axes[1]:
        ax.set_xlabel("training iteration")
    g = a[a.arm == "GRPO_K0"].set_index("iteration")
    end_label(axes[1, 0], 10, 100 * g.loc[10, "degenerate_among_worst"], "worst", dy=2)
    end_label(axes[1, 0], 10, 100 * g.loc[10, "degenerate_among_best"], "best", dy=-2)
    axes[0, 1].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right")
    fig.tight_layout()
    heading(fig, "Leaked replies win and lose at the base rate; degenerate replies lose",
            "Share of each round's best and worst candidates (ties averaged) that carry the artefact, "
            "train rounds with at least two distinct rewards.")
    fig.savefig(os.path.join(FIG, "artefacts.png"))
    plt.close(fig)


def fig_lose_end():
    m = pd.read_csv(os.path.join(TAB, "marker_share_top_sentences.csv"))
    pool = pd.read_csv(os.path.join(TAB, "sentence_pool.csv"))
    pool_rate = 100 * pool.has_marker.mean()
    variants = [("all", "all candidates"), ("no_leak", "leaked removed"), ("clean", "leaked and degenerate\nremoved")]
    mean = m.groupby(["variant", "arm"]).marker_share_top10_lose.mean().mul(100)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w, x = 0.3, np.arange(len(variants))
    for i, arm in enumerate(ARMS):
        vals = [mean[(v, arm)] for v, _ in variants]
        pos = x + (i - 0.5) * (w + 0.04)
        ax.bar(pos, vals, w, color=COL[arm], label=LAB[arm], edgecolor=SURFACE, linewidth=1)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(pool_rate, color=INK2, lw=1, ls=(0, (4, 3)), label=f"whole pool ({pool_rate:.1f}%)")
    ax.set_xticks(x, [n for _, n in variants])
    ax.set_xlim(-0.55, len(variants) - 0.45)
    ax.set_ylim(0, 58)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("sentences with a chat marker")
    ax.legend(loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()
    heading(fig, "Chat markers crowd the lose end until degenerate replies are removed",
            "Marker share among each direction's 10 lowest-projecting pool sentences, mean over iterations 1–10.\n"
            "The 10 highest-projecting sentences carry none in any variant.")
    fig.savefig(os.path.join(FIG, "lose_end_markers.png"))
    plt.close(fig)


def fig_cosines():
    c = pd.read_csv(os.path.join(TAB, "cosines.csv"))
    c = c[c.variant == "all"]
    k0, k5 = (c[c.arm == arm].set_index("iteration") for arm in ARMS)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), sharex=True, sharey=True)
    (a1, a2), (a3, a4) = axes

    line(a1, ITERS, k0.k0_vs_k5_cos, PAIR, ls="--", hollow=True)
    line(a1, ITERS, k0.k0_vs_k5_corrected, PAIR)
    a1.set_title("K=0 direction vs K=5 direction, same iteration")
    end_label(a1, 10, k0.loc[10, "k0_vs_k5_corrected"], "corrected", dy=3)
    end_label(a1, 10, k0.loc[10, "k0_vs_k5_cos"], "raw", dy=-3)

    for arm, g in zip(ARMS, (k0, k5)):
        line(a2, ITERS, g.reliability, COL[arm], marker=MARK[arm])
        end_label(a2, 10, g.loc[10, "reliability"], LAB[arm])
    a2.set_title("Reliability (split-half cosine of a direction)")

    for ax, arm, g in ((a3, "GRPO_K0", k0), (a4, "GRPO_K5", k5)):
        s = g.loc[2:]
        line(ax, s.index, s.stability_cos, COL[arm], ls="--", marker=MARK[arm], hollow=True)
        line(ax, s.index, s.stability_corrected, COL[arm], marker=MARK[arm])
        ax.set_title(f"{LAB[arm]}: iteration t vs iteration t−1")
        ax.set_xlabel("training iteration")
    end_label(a4, 10, k5.loc[10, "stability_corrected"], "corrected")
    end_label(a4, 10, k5.loc[10, "stability_cos"], "raw")

    for ax in (a1, a4):
        ax.axvspan(8.5, 10.4, color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.text(9.45, -0.4, "K=5 reliability\n≤ 0.17", ha="center", va="bottom", fontsize=7.5, color=INK2)
    for ax in axes.flat:
        iter_axis(ax)
        ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
        ax.set_ylim(-0.45, 1.12)
    a1.set_ylabel("cosine")
    a3.set_ylabel("cosine")
    handles = [Line2D([], [], color=INK2, lw=LW, marker="o", ms=MS, mec=SURFACE, label="noise-corrected"),
               Line2D([], [], color=INK2, lw=LW, ls="--", marker="o", ms=MS, mfc=SURFACE, mec=INK2, mew=1.4,
                      label="raw")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.035))
    fig.tight_layout()
    heading(fig, "Most of the late divergence is noise: K=5 stops having a direction",
            "Directions from all candidates. Corrected = cross-half cosine / √(reliability₁ × reliability₂), "
            "50 random patient half-splits.")
    fig.savefig(os.path.join(FIG, "cosines.png"))
    plt.close(fig)


def fig_features():
    f = pd.read_csv(os.path.join(TAB, "k0_minus_k5_features.csv"))
    f = f[f.variant == "clean"]
    order = ["praise phrase", "'you are ...' affirmation", "'we' / 'together'", "reflection opener",
             "asks a question", "advice / tips", "numbered or bulleted list", "cut off mid-sentence",
             "length (log chars)"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    vmax = 0.8
    for ax, arm in zip(axes, ARMS):
        p = f[f.arm == arm].pivot(index="feature", columns="iteration", values="corr").loc[order, ITERS]
        im = ax.pcolormesh(p.values, cmap=DIVERGE, vmin=-vmax, vmax=vmax, edgecolors=SURFACE, linewidth=1.5)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00"),
                        ha="center", va="center", fontsize=7.2, color="#ffffff" if abs(v) > 0.45 else INK)
        ax.set_xticks(np.arange(len(ITERS)) + 0.5, ITERS)
        ax.set_yticks(np.arange(len(order)) + 0.5, order)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"within {LAB[arm]}'s candidates")
        ax.set_xlabel("training iteration")
    axes[0].invert_yaxis()  # shared y: invert once, not per axis
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2)
    cb.set_label("← K=5 side      K=0 side →", color=INK2)
    heading(fig, "Praise lines up with the K=0 side from iteration 5, mostly within K=0's own replies",
            "Correlation of each text feature with a candidate's projection on the K0 − K5 direction "
            "(leaked and degenerate candidates removed). Length is the strongest correlate.", y=0.95)
    fig.savefig(os.path.join(FIG, "features.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    style()
    fig_artefacts()
    fig_lose_end()
    fig_cosines()
    fig_features()
    print("wrote", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
