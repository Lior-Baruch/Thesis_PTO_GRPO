"""Draw Figure 1, the GRPO-group schematic, at ACL full-text width.

The EDA's hand-authored schematic (``eda/results/schematics/grpo_group_rollout.png``) is a
portrait, three-column diagram sized for a slide; placed in one ACL column its box text prints at
about 4 pt. This redraws the same content as a left-to-right pipeline sized for ``figure*`` at
``\\textwidth`` (6.3 in), so every label prints at 6-7 pt. Nothing here reads data: it is a
diagram of the method as ``sec:method`` states it (G = 8 completions per prompt, a K-turn rollout
with the patient first, one oracle call per candidate, group-standardised advantages, a
PPO-clipped step with a KL penalty).

    & ..\\..\\.venv\\Scripts\\python.exe render_schematic.py

Writes ``figures/method_grpo_group.png`` (the destination name the .tex references, which
``sync_figures.py`` used to copy from the EDA tree and no longer lists).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "method_grpo_group.png"

NAVY = "#1F3A5F"
GREY = "#5A5A5A"
# (fill, edge) per role -- the EDA schematic's palette: green = data produced, purple = an API
# model we call, blue = the policy being trained, orange = the oracle, yellow = the update.
ROLE = {
    "source": ("#EAF1E7", "#5B8C5A"),
    "data": ("#DCEDE3", "#008A63"),
    "api": ("#EDE6F3", "#7B5EA7"),
    "policy": ("#DCEBF5", "#0072B2"),
    "oracle": ("#FBE8D5", "#D55E00"),
    "neutral": ("#EFF2F6", "#8A96A3"),
    "update": ("#FFF4E0", "#B8860B"),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def node(ax, x, y, w, h, text, role="neutral", fs=7.0, bold=False, lw=1.0):
    fill, edge = ROLE[role]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={min(w, h) * 0.18:.3f}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#22282F", zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, a, b, color=NAVY, lw=0.9, side="h", scale=7):
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "h":
        sx, sy = ax0 + aw / 2, ay0
        ex, ey = bx0 - bw / 2, by0
    else:
        sx, sy = ax0, ay0 - ah / 2
        ex, ey = bx0, by0 + bh / 2
    ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=scale,
                                 color=color, linewidth=lw, zorder=1.5, shrinkA=1.0, shrinkB=1.0))


def main() -> int:
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    rows = (32.0, 23.0, 12.0)            # completion 1, completion 2, completion G
    # -- prompt ------------------------------------------------------------------------------
    prompt = node(ax, 9.0, 23.0, 17.2, 11.0,
                  "prompt $c$\nprefix, $\\geq$ MCL $= 12$\nutterances, ending\non a patient turn",
                  role="source", fs=5.9)
    # -- the group ---------------------------------------------------------------------------
    ax.text(24.6, 39.4, "$\\pi_n$ samples $G{=}8$\ncompletions", ha="center", va="center",
            fontsize=6.6, color=NAVY, fontweight="bold", linespacing=1.2)
    comps = [node(ax, 24.6, y, 12.2, 5.0, f"completion $t_{{{lab}}}$", role="data", fs=6.4)
             for y, lab in zip(rows, ("1", "2", "G"))]
    ax.text(24.6, 17.5, "$\\vdots$", ha="center", va="center", fontsize=9, color=GREY)
    for c in comps:
        arrow(ax, prompt, c)
    # -- the rollout -------------------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((32.6, 8.2), 31.4, 27.8, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="none", edgecolor="#B8BFC8", linewidth=0.8,
                                linestyle=(0, (3, 2)), zorder=0.5))
    ax.text(49.0, 39.4, "look-ahead rollout $\\tau_K$:\n$K{=}5$ further turns, patient first",
            ha="center", va="center", fontsize=6.6, color=NAVY, fontweight="bold", linespacing=1.2)
    xs = (36.4, 42.4, 48.4, 54.4, 60.4)
    chains = []
    for y, c in zip(rows, comps):
        chain = []
        for i, x in enumerate(xs):
            is_p = i % 2 == 0
            chain.append(node(ax, x, y, 4.8, 4.6, "$P$" if is_p else "$\\pi_n$",
                              role="api" if is_p else "policy", fs=6.6))
        arrow(ax, c, chain[0])
        for a, b in zip(chain, chain[1:]):
            arrow(ax, a, b, scale=5)
        chains.append(chain)
    ax.text(48.4, 17.5, "$\\vdots$", ha="center", va="center", fontsize=9, color=GREY)
    ax.text(48.4, 5.0, "$K = 0$: the rollout is skipped and the oracle scores $c \\oplus t_g$ alone (standard GRPO)",
            ha="center", va="center", fontsize=6.2, color=NAVY, fontstyle="italic")
    # -- the oracle --------------------------------------------------------------------------
    oracle = node(ax, 72.4, 23.0, 14.8, 11.0,
                  "oracle $O$ scores\n$c \\oplus t_g \\oplus \\tau_K(c \\oplus t_g)$\non Q1+Q2 $\\rightarrow r_g$",
                  role="oracle", fs=6.1)
    for chain in chains:
        arrow(ax, chain[-1], oracle)
    # -- the update --------------------------------------------------------------------------
    adv = node(ax, 90.6, 31.5, 17.6, 8.0,
               "group-relative advantage\n$A_g = (r_g - \\bar r)\\,/\\,\\sigma_r$\nover the $G$ siblings",
               role="neutral", fs=5.9)
    upd = node(ax, 90.6, 14.0, 17.6, 11.0,
               "PPO-clipped policy-\ngradient step on all $G$:\n$\\sum_g A_g\\,\\nabla \\log \\pi(t_g \\mid c)$\n$+\\ \\beta\\,\\mathrm{KL}(\\pi \\,\\|\\, \\pi_{n})$",
               role="update", fs=5.9)
    arrow(ax, oracle, adv)
    arrow(ax, adv, upd, side="v")
    # -- legend ------------------------------------------------------------------------------
    ax.text(1.0, 1.0, "green: transcripts produced    purple: patient simulator $P$ (API)    "
                      "blue: the policy $\\pi_n$ being trained    orange: oracle $O$    yellow: the update",
            ha="left", va="bottom", fontsize=5.9, color=GREY)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
