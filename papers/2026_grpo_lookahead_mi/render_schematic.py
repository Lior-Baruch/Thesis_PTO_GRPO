"""Draw Figure 1, the GRPO-group schematic, at the exact width the paper includes it.

The EDA's hand-authored schematic (``eda/results/schematics/grpo_group_rollout.png``) is a
portrait, three-column diagram sized for a slide; placed in one ACL column its box text prints at
about 4 pt. This redraws the same content as a left-to-right pipeline for a ``figure*``. Nothing
here reads data: it is a diagram of the method as ``sec:method`` states it (G = 8 completions per
prompt, a K-turn rollout with the patient first, one oracle call per candidate, group-standardised
advantages, the policy step of Eq. 2 with its KL penalty to pi_n).

SIZING (2026-10-01): the canvas is the include width read from ``sections/03_method.tex``
(``0.82\\textwidth`` today) minus the save pad, the axes fill it, and the data units are isotropic,
so the PNG is exactly as wide as the page prints it and every point size below is a print size
(nothing under 5.8 pt; mathtext subscripts are smaller by construction). Until then the canvas
was 6.3 in, the PNG came out 5.70 in and printed at 0.91 of its sizes (the smallest box text at
5.0 pt). ``main()`` fails if a box text does not fit its box, two texts overlap, or the PNG width
misses the include width.

    & ..\\..\\.venv\\Scripts\\python.exe render_schematic.py

Writes ``figures/method_grpo_group.png`` (the destination name the .tex references, which
``sync_figures.py`` used to copy from the EDA tree and no longer lists).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "method_grpo_group.png"
TEX = HERE / "sections" / "03_method.tex"
# ACL \textwidth, read off main.log ("* \textwidth=455.24411pt", TeX points).
TEXTWIDTH_IN = 455.24411 / 72.27
PAD_IN = 0.03                      # savefig pad around the axes, inside the include width
DPI = 300
XMAX, YMAX = 100.0, 33.2           # data extent; 1 unit = the same length on both axes


def include_width_in() -> float:
    """The width 03_method.tex includes the PNG at, in inches."""
    m = re.search(r"\\includegraphics\[width=([0-9.]+)\\textwidth\]\{figures/method_grpo_group\.png\}",
                  TEX.read_text(encoding="utf-8"))
    if m is None:
        raise SystemExit(f"{TEX.name} does not include figures/method_grpo_group.png at a \\textwidth fraction")
    return float(m.group(1)) * TEXTWIDTH_IN

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


BOXED = []                         # (text, patch) pairs that main() checks for fit


def node(ax, x, y, w, h, text, role="neutral", fs=7.0, bold=False, lw=1.0):
    fill, edge = ROLE[role]
    patch = ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={min(w, h) * 0.18:.3f}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    label = ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#22282F", zorder=3,
                    fontweight="bold" if bold else "normal", linespacing=1.3)
    BOXED.append((label, patch))
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


def _overlap(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def check_layout(fig, free, dashed) -> list[str]:
    """Every box text inside its box with >= 1 pt to spare, no two texts overlapping, no free
    text (headers, note, legend) on a box, and nothing outside the canvas."""
    r = fig.canvas.get_renderer()
    pt = DPI / 72
    probs = []
    for label, patch in BOXED:
        t, p = label.get_window_extent(r), patch.get_window_extent(r)
        if not (t.x0 >= p.x0 + pt and t.x1 <= p.x1 - pt and t.y0 >= p.y0 + pt and t.y1 <= p.y1 - pt):
            probs.append(f"text {label.get_text()!r} does not fit its box")
    texts = [lab for lab, _ in BOXED] + list(free)
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if _overlap(a.get_window_extent(r), b.get_window_extent(r)):
                probs.append(f"texts overlap: {a.get_text()[:30]!r} / {b.get_text()[:30]!r}")
    for t in free:
        if t.get_text() == "$\\vdots$":
            continue
        for patch in [p for _, p in BOXED] + [dashed]:
            if _overlap(t.get_window_extent(r), patch.get_window_extent(r)):
                probs.append(f"text {t.get_text()[:30]!r} sits on a box")
    canvas = fig.bbox
    for t in texts:
        e = t.get_window_extent(r)
        if e.x0 < canvas.x0 or e.x1 > canvas.x1 or e.y0 < canvas.y0 or e.y1 > canvas.y1:
            probs.append(f"text {t.get_text()[:30]!r} runs off the canvas")
    return probs


def main() -> int:
    width = include_width_in() - 2 * PAD_IN         # the PNG is the canvas plus the pad
    fig = plt.figure(figsize=(width, width * YMAX / XMAX), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    BOXED.clear()
    free = []

    rows = (25.1, 18.5, 11.3)            # completion 1, completion 2, completion G
    # -- prompt ------------------------------------------------------------------------------
    prompt = node(ax, 8.4, 18.5, 16.0, 10.6,
                  "prompt $c$\nprefix of $\\geq 12$\nutterances (MCL),\nending on a\npatient turn",
                  role="source", fs=6.0)
    # -- the group ---------------------------------------------------------------------------
    # Centred over the prompt-to-completion arrows: over the completions it met the rollout's
    # header once both print at 6.2 pt.
    free.append(ax.text(18.5, 30.85, "$\\pi$ samples\n$G{=}8$ completions", ha="center",
                        va="center", fontsize=6.2, color=NAVY, fontweight="bold", linespacing=1.2))
    comps = [node(ax, 21.6, y, 5.4, 3.8, f"$t_{{{lab}}}$", role="data", fs=7.0)
             for y, lab in zip(rows, ("1", "2", "G"))]
    free.append(ax.text(21.6, 14.9, "$\\vdots$", ha="center", va="center", fontsize=9, color=GREY))
    for c in comps:
        arrow(ax, prompt, c)
    # -- the rollout -------------------------------------------------------------------------
    dashed = ax.add_patch(FancyBboxPatch((27.0, 8.9), 28.4, 18.6,
                                         boxstyle="round,pad=0.3,rounding_size=1.2",
                                         facecolor="none", edgecolor="#B8BFC8", linewidth=0.8,
                                         linestyle=(0, (3, 2)), zorder=0.5))
    free.append(ax.text(41.2, 30.85, "look-ahead rollout $\\tau_K$:\n$K{=}5$ turns, patient first",
                        ha="center", va="center", fontsize=6.2, color=NAVY, fontweight="bold",
                        linespacing=1.2))
    xs = (30.0, 35.6, 41.2, 46.8, 52.4)
    chains = []
    for y, c in zip(rows, comps):
        chain = []
        for i, x in enumerate(xs):
            is_p = i % 2 == 0
            chain.append(node(ax, x, y, 4.0, 3.6, "$P$" if is_p else "$\\pi$",
                              role="api" if is_p else "policy", fs=6.6))
        arrow(ax, c, chain[0])
        for a, b in zip(chain, chain[1:]):
            arrow(ax, a, b, scale=5)
        chains.append(chain)
    free.append(ax.text(41.2, 14.9, "$\\vdots$", ha="center", va="center", fontsize=9, color=GREY))
    free.append(ax.text(38.0, 6.25, "$K = 0$: the rollout is skipped and the oracle scores "
                                     "$c \\oplus t_g$ alone (standard GRPO)",
                        ha="center", va="center", fontsize=6.2, color=NAVY, fontstyle="italic"))
    # -- the oracle --------------------------------------------------------------------------
    oracle = node(ax, 65.55, 18.5, 15.0, 7.3,
                  "oracle $O$ scores\n$c \\oplus t_g \\oplus \\tau_K(c \\oplus t_g)$\non Q1+Q2 $\\rightarrow r_g$",
                  role="oracle", fs=6.0)
    for chain in chains:
        arrow(ax, chain[-1], oracle)
    # -- the update (2026-10-01: "within the group", and the step said in words, as Eq. 2 is a
    # per-token clipped surrogate rather than the sum the box used to print) ------------------
    adv = node(ax, 87.725, 28.4, 23.75, 6.9,
               "group-relative advantage\n$A_g = (r_g - \\bar r)\\,/\\,\\sigma_r$\nwithin the group",
               role="neutral", fs=6.0)
    upd = node(ax, 87.725, 16.95, 23.75, 9.0,
               "policy step (Eq. 2):\nevery completion\nweighted by its $A_g$,\nKL penalty to $\\pi_n$",
               role="update", fs=6.0)
    arrow(ax, oracle, adv)
    arrow(ax, adv, upd, side="v")
    # -- legend (two lines since 2026-10-01: on one it was wider than the figure) ------------
    free.append(ax.text(0.4, 0.25, "green: transcripts produced    purple: patient simulator $P$ (API)    "
                                   "blue: the policy $\\pi$ being trained\n"
                                   "orange: oracle $O$    yellow: the update",
                        ha="left", va="bottom", fontsize=6.0, color=GREY))

    fig.canvas.draw()
    probs = check_layout(fig, free, dashed)
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    if abs(tight.width - width) > 0.005 * width:
        probs.append(f"content is {tight.width:.3f} in wide, the canvas {width:.3f} in")
    if probs:
        plt.close(fig)
        raise SystemExit("schematic layout problems:\n  " + "\n  ".join(probs))

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", pad_inches=PAD_IN, facecolor="white")
    plt.close(fig)
    got = plt.imread(OUT).shape[1] / DPI
    if abs(got - include_width_in()) > 0.01 * include_width_in():
        raise SystemExit(f"{OUT.name} is {got:.3f} in wide; the .tex includes it at {include_width_in():.3f} in")
    print(f"wrote {OUT} ({got:.3f} in wide; included at {include_width_in():.3f} in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
