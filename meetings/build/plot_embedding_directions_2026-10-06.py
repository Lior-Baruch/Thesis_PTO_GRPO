"""
plot_embedding_directions_2026-10-06.py -- summary figure of embedding_directions_2026-10-06.py.

Reads ``tables/categories.csv`` (clean candidates, Doron's win-lose estimator) and draws, for the
sentence categories the paper's behaviour claims rest on (praise, question, advice), how far each
run's within-round reward direction points toward that category's pool sentences, by training
iteration: the mean over the five embedders (line) and their range (band). K=0 / K=5 in the paper's
colours, with marker and line style as the second encoding.

    .venv/Scripts/python.exe meetings/build/plot_embedding_directions_2026-10-06.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "meetings", "2026-10-06_embedding_directions")
COL = {"GRPO_K0": "#d55e00", "GRPO_K5": "#e69f00"}
STY = {"GRPO_K0": dict(marker="o", ls="-"), "GRPO_K5": dict(marker="s", ls="--")}
LAB = {"GRPO_K0": "K=0", "GRPO_K5": "K=5"}
CATS = ["praise", "question", "advice"]


def main():
    c = pd.read_csv(os.path.join(OUT, "tables", "categories.csv"))
    c = c[(c.variant == "clean") & (c.estimator == "winlose") & c.side.isin(COL)]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(CATS), figsize=(7.2, 2.3), sharey=True)
    rows = []
    for ax, cat in zip(axes, CATS):
        for side in COL:
            g = c[(c.category == cat) & (c.side == side)].groupby("iteration").z
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(m.index, lo, hi, color=COL[side], alpha=0.18, lw=0)
            ax.plot(m.index, m, color=COL[side], ms=3.5, lw=1.4, label=LAB[side], **STY[side])
            rows += [dict(category=cat, arm=side, iteration=int(i), mean_z=round(float(m[i]), 3),
                          min_z=round(float(lo[i]), 3), max_z=round(float(hi[i]), 3)) for i in m.index]
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_title(cat, loc="left", fontweight="bold")
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xlabel("training iteration")
        ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    axes[0].set_ylabel("pull toward the category\n(z, category vs other sentences)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "categories_summary.png"), dpi=220)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tables", "categories_summary.csv"), index=False)
    print("wrote", os.path.join(OUT, "figures", "categories_summary.png"))


if __name__ == "__main__":
    main()
