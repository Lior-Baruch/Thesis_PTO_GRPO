"""Draw the README's results figure (light and dark variants) from the EDA's tracked tables.

    .venv\\Scripts\\python.exe docs\\make_readme_figure.py

Writes ``docs/assets/results_light.png`` and ``docs/assets/results_dark.png``. Nothing here computes
a number: every value is read from
``Exp3_PTO_GRPO/eda/results/lookahead/shared_base/tables/shared_base.xlsx``

- (a) sheet ``k_contrast``, GRPO, iteration 10: ``dz_K5_minus_K0`` per instrument and judge (the
  persona-paired effect of look-ahead; MICI is lower-is-better, so its sign is flipped to read
  "positive favors K=5" like the rest). These are the d_z columns of the paper's Table 1.
- (b) sheet ``process_levels_gpt-4o-mini``: ``th_PRA_rate`` and ``th_CR_rate`` (share of therapist
  turns whose dominant function is non-specific praise / complex reflection, training oracle) at the
  Base and at iteration 10 of each run. These are rows of the paper's Table 3.

Colors: the reference categorical palette's first two slots (orange K=0, blue K=5), validated for
both modes with the dataviz skill's ``validate_palette.js``; the Base is a neutral gray.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "Exp3_PTO_GRPO" / "eda" / "results" / "lookahead" / "shared_base" / "tables" / "shared_base.xlsx"
OUT = Path(__file__).resolve().parent / "assets"

INSTRUMENTS = [("Q1", "Q1 (reward)"), ("Q2", "Q2 (reward)"), ("MITI", "MITI"), ("WAI-SR", "WAI-SR"),
               ("CSQ-8", "CSQ-8"), ("MI-SAT", "MI-SAT"), ("PCT", "Patient change talk"),
               ("MICI", "fewer MI-inconsistent acts")]
JUDGES = [("gpt-4o-mini", "training judge (gpt-4o-mini)"), ("claude-haiku-4-5", "held-out judge (Claude Haiku 4.5)")]

THEMES = {
    "light": dict(surface="#ffffff", ink="#1f2328", ink2="#59636e", grid="#d1d9e0", zero="#59636e",
                  k0="#eb6834", k5="#2a78d6", base="#a3a29c"),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", ink2="#9198a1", grid="#3d444d", zero="#9198a1",
                 k0="#d95926", k5="#3987e5", base="#6e6d68"),
}


def load():
    k = pd.read_excel(XLSX, sheet_name="k_contrast")
    k = k[(k.method == "GRPO") & (k.iteration == 10)]
    dz = {}
    for j, _ in JUDGES:
        for m, _ in INSTRUMENTS:
            v = float(k[(k.judge == j) & (k.metric == m)]["dz_K5_minus_K0"].iloc[0])
            dz[(j, m)] = -v if m == "MICI" else v          # lower-is-better: flip to "favors K=5"
    p = pd.read_excel(XLSX, sheet_name="process_levels_gpt-4o-mini")
    beh = {}
    for code, col in (("praise", "th_PRA_rate"), ("reflect", "th_CR_rate")):
        beh[(code, "Base")] = float(p[(p.arm == "GRPO_LA0") & (p.iteration == 0)][col].iloc[0])
        beh[(code, "K=0")] = float(p[(p.arm == "GRPO_LA0") & (p.iteration == 10)][col].iloc[0])
        beh[(code, "K=5")] = float(p[(p.arm == "GRPO_LA5") & (p.iteration == 10)][col].iloc[0])
    return dz, beh


def draw(theme: str, dz, beh) -> Path:
    t = THEMES[theme]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                         "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
                         "ytick.color": t["ink"], "axes.edgecolor": t["grid"]})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(t["surface"])
    for ax in (a, b):
        ax.set_facecolor(t["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # (a) effect of look-ahead on every instrument, both judges
    y = np.arange(len(INSTRUMENTS))[::-1]
    a.axvline(0, color=t["zero"], lw=1.0)
    for (j, jlab), off, filled in zip(JUDGES, (0.14, -0.14), (True, False)):
        xs = [dz[(j, m)] for m, _ in INSTRUMENTS]
        a.hlines(y + off, 0, xs, color=t["k5"], lw=2, alpha=0.55)
        a.scatter(xs, y + off, s=64, zorder=3, color=t["k5"] if filled else t["surface"],
                  edgecolor=t["k5"], linewidth=2, label=jlab)
    a.set_yticks(y)
    a.set_yticklabels([lab for _, lab in INSTRUMENTS])
    a.set_xlim(-0.1, 2.6)
    a.set_xlabel("effect of look-ahead, Cohen's $d_z$  (K=5 − K=0; > 0 favors K=5)")
    a.grid(axis="x", color=t["grid"], lw=0.8)
    a.set_axisbelow(True)
    a.set_title("(a) K=5 beats K=0 on all 8 instruments, iteration 10", loc="left", fontweight="bold")
    leg = a.legend(loc="center right", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # (b) what each policy learned to do
    groups = [("praise", "non-specific praise"), ("reflect", "complex reflection")]
    states = [("Base", t["base"], "untrained model (Base)"), ("K=0", t["k0"], "K=0 policy, iteration 10"),
              ("K=5", t["k5"], "K=5 policy, iteration 10")]
    w = 0.26
    x = np.arange(len(groups))
    for i, (st, col, lab) in enumerate(states):
        vals = [beh[(g, st)] for g, _ in groups]
        bars = b.bar(x + (i - 1) * w, vals, width=w - 0.03, color=col, label=lab,
                     zorder=2)
        for bar, v in zip(bars, vals):
            b.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                   fontsize=10, color=t["ink2"])
    b.set_xticks(x)
    b.set_xticklabels([lab for _, lab in groups])
    b.set_ylim(0, 0.47)
    b.set_ylabel("share of therapist turns (training judge)")
    b.grid(axis="y", color=t["grid"], lw=0.8)
    b.set_axisbelow(True)
    b.set_title("(b) K=0 praises, K=5 reflects", loc="left", fontweight="bold")
    leg = b.legend(loc="upper right", frameon=False, handlelength=1.0, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout(w_pad=2.5)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{theme}.png"
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return out


def main() -> None:
    dz, beh = load()
    for theme in THEMES:
        print("wrote", draw(theme, dz, beh))


if __name__ == "__main__":
    main()
