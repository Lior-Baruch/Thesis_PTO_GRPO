"""Draw Figure 1, the GRPO-group schematic, at the exact width the paper includes it.

The EDA's hand-authored schematic (``eda/results/schematics/grpo_group_rollout.png``) is a
portrait, three-column diagram sized for a slide; placed in one ACL column its box text prints at
about 4 pt. This redraws the same content for the paper. Nothing here reads data: it is a diagram
of the method as ``sec:method`` states it (G = 8 completions per prompt, a K-turn rollout with the
patient first, one oracle call per candidate, group-standardised advantages, the policy step of
Eq. 2 with its KL penalty to pi_n).

LAYOUT (2026-10-07): one column (``0.48\\textwidth`` in ``sections/03_method.tex``), drawn as a
U. Top band: the prompt box at left feeds the G rows, each a completion t_g and its 5-node rollout
chain inside the dashed box; the chain ends join a bus on the right that drops into the oracle.
Bottom band, right to left: oracle -> group-relative advantage -> policy step. The K = 0 note sits
between the bands, under the rollout it qualifies. The colour legend is gone (the labels and the
caption name every role). Until 2026-10-07 this was a left-to-right ``figure*`` at
``0.82\\textwidth``.

SIZING: the canvas is the include width read from ``03_method.tex`` minus the save pad, the axes
fill it, and the data units are isotropic. ``XMAX`` is that canvas in points at ``0.48\\textwidth``,
so at that width 1 data unit = 1 pt and every point size below is a print size (nothing under
``MIN_FS``; mathtext subscripts are smaller by construction). ``main()`` fails if a box text does
not fit its box, two texts overlap, a text is under ``MIN_FS``, or the PNG width misses the include
width.

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
MIN_FS = 6.5                       # smallest font size drawn, in pt
# Data extent: XMAX is the canvas width in pt at 0.48\textwidth (3.024 in - 2 x 0.03 in pad), so
# 1 data unit = 1 pt there; 1 unit = the same length on both axes.
XMAX, YMAX = 213.4, 130.5


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
BOX_PAD = 1.2                      # FancyBboxPatch pad, data units (pt); node sizes include it

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


BOXED = []                         # (text, patch) pairs that main() checks for fit


def node(ax, x, y, w, h, text, role="neutral", fs=7.0, lw=0.9, ls=1.05):
    """A rounded box of OUTER size w x h (pad included) centred on (x, y), with centred text."""
    fill, edge = ROLE[role]
    patch = ax.add_patch(FancyBboxPatch(
        (x - w / 2 + BOX_PAD, y - h / 2 + BOX_PAD), w - 2 * BOX_PAD, h - 2 * BOX_PAD,
        boxstyle=f"round,pad={BOX_PAD},rounding_size={min(min(w, h) * 0.16, 3.0):.3f}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    label = ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#22282F", zorder=3,
                    linespacing=ls)
    BOXED.append((label, patch))
    return (x, y, w, h)


def arrow_xy(ax, start, end, color=NAVY, lw=0.8, scale=5.5, shrink_a=0.8, shrink_b=0.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale, color=color,
                                 linewidth=lw, zorder=1.5, shrinkA=shrink_a, shrinkB=shrink_b))


def arrow(ax, a, b, side="r", **kw):
    """Box a -> box b: 'r' a's right edge to b's left edge, 'l' a's left edge to b's right edge."""
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "r":
        arrow_xy(ax, (ax0 + aw / 2, ay0), (bx0 - bw / 2, by0), **kw)
    else:
        arrow_xy(ax, (ax0 - aw / 2, ay0), (bx0 + bw / 2, by0), **kw)


def _overlap(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def check_layout(fig, free, dashed) -> list[str]:
    """Every box text inside its box with >= 1 pt to spare, no two texts overlapping, no free
    text (headers, note) on a box, no text under MIN_FS, and nothing outside the canvas."""
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
    for t in texts:
        if t.get_fontsize() < MIN_FS:
            probs.append(f"text {t.get_text()[:30]!r} is {t.get_fontsize()} pt (< {MIN_FS})")
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
    margin = 0.5                                     # canvas edge to the outermost boxes
    x_left, x_right = margin, XMAX - margin

    # -- vertical bands (bottom up, pt) --------------------------------------------------------
    bot_h = 33.0                                     # bottom band: oracle, advantage, policy step
    bot_y = margin + bot_h / 2
    note_y = margin + bot_h + 10.75                  # the K = 0 note, between the bands
    node_h = 13.0
    rows = (103.0, 86.0, 64.0)                       # completion 1, completion 2, completion G
    dash_pad = 3.0
    head_y = rows[0] + node_h / 2 + dash_pad + 9.5   # the two bold headers

    # -- horizontal columns (pt) ---------------------------------------------------------------
    prompt_w = 54.0
    t_w = 17.0
    t_x = x_left + prompt_w + 10.0 + t_w / 2
    chain_w, chain_gap = 15.0, 9.0
    xs = [t_x + t_w / 2 + 9.0 + chain_w / 2 + i * (chain_w + chain_gap) for i in range(5)]
    bus_x = xs[-1] + chain_w / 2 + 6.5

    # -- prompt --------------------------------------------------------------------------------
    prompt_top, prompt_bot = rows[0] + node_h / 2, rows[-1] - node_h / 2
    prompt = node(ax, x_left + prompt_w / 2, (prompt_top + prompt_bot) / 2, prompt_w,
                  prompt_top - prompt_bot,
                  "prompt $c$\nprefix of $\\geq 12$\nutterances\n(MCL), ending\non a patient\nturn",
                  role="source", fs=7.0, ls=1.1)
    # -- the group -----------------------------------------------------------------------------
    free.append(ax.text((x_left + t_x + t_w / 2) / 2, head_y, "$\\pi$ samples\n$G{=}8$ completions",
                        ha="center", va="center", fontsize=7.0, color=NAVY, fontweight="bold",
                        linespacing=1.1))
    comps = [node(ax, t_x, y, t_w, node_h, f"$t_{{{lab}}}$", role="data", fs=7.5)
             for y, lab in zip(rows, ("1", "2", "G"))]
    vdots_y = (rows[1] + rows[2]) / 2
    free.append(ax.text(t_x, vdots_y, "$\\vdots$", ha="center", va="center", fontsize=8, color=GREY))
    for y in rows:
        arrow_xy(ax, (prompt[0] + prompt[2] / 2, y), (t_x - t_w / 2, y))
    # -- the rollout ---------------------------------------------------------------------------
    d_x0 = xs[0] - chain_w / 2 - dash_pad
    d_x1 = xs[-1] + chain_w / 2 + dash_pad
    d_y0 = rows[-1] - node_h / 2 - dash_pad
    d_y1 = rows[0] + node_h / 2 + dash_pad
    dashed = ax.add_patch(FancyBboxPatch((d_x0 + 0.6, d_y0 + 0.6), d_x1 - d_x0 - 1.2, d_y1 - d_y0 - 1.2,
                                         boxstyle="round,pad=0.6,rounding_size=3.0",
                                         facecolor="none", edgecolor="#B8BFC8", linewidth=0.7,
                                         linestyle=(0, (3, 2)), zorder=0.5))
    free.append(ax.text((d_x0 + d_x1) / 2, head_y, "look-ahead rollout $\\tau_K$:\n$K{=}5$ turns, patient first",
                        ha="center", va="center", fontsize=7.0, color=NAVY, fontweight="bold",
                        linespacing=1.1))
    chains = []
    for y, c in zip(rows, comps):
        chain = []
        for i, x in enumerate(xs):
            is_p = i % 2 == 0
            chain.append(node(ax, x, y, chain_w, node_h, "$P$" if is_p else "$\\pi$",
                              role="api" if is_p else "policy", fs=7.0))
        arrow(ax, c, chain[0])
        for a, b in zip(chain, chain[1:]):
            arrow(ax, a, b)
        chains.append(chain)
    free.append(ax.text(xs[2], vdots_y, "$\\vdots$", ha="center", va="center", fontsize=8, color=GREY))
    free.append(ax.text((x_left + bus_x) / 2 - 3.0, note_y,
                        "$K = 0$: the rollout is skipped and the oracle scores\n"
                        "$c \\oplus t_g$ alone (standard GRPO)",
                        ha="center", va="center", fontsize=6.5, color=NAVY, fontstyle="italic",
                        linespacing=1.1))
    # -- bottom band, right to left: oracle -> advantage -> policy step ------------------------
    oracle_w, adv_w, upd_w = 61.5, 61.0, 70.0
    oracle = node(ax, x_right - oracle_w / 2, bot_y, oracle_w, bot_h,
                  "oracle $O$ scores\n$c \\oplus t_g \\oplus \\tau_K(c \\oplus t_g)$\non Q1+Q2 $\\rightarrow r_g$",
                  role="oracle", fs=6.5, ls=1.1)
    upd = node(ax, x_left + upd_w / 2, bot_y, upd_w, bot_h,
               "policy step (Eq. 2):\nevery completion\nweighted by its $A_g$,\nKL penalty to $\\pi_n$",
               role="update", fs=6.5)
    adv_x = ((x_left + upd_w) + (x_right - oracle_w)) / 2
    adv = node(ax, adv_x, bot_y, adv_w, bot_h,
               "group-relative\nadvantage\n$A_g = (r_g - \\bar r)\\,/\\,\\sigma_r$\nwithin the group",
               role="neutral", fs=6.5)
    arrow(ax, oracle, adv, side="l")
    arrow(ax, adv, upd, side="l")
    # -- the bus: every chain end joins one line that drops into the oracle ------------------
    for chain in chains:
        end = chain[-1]
        ax.plot([end[0] + end[2] / 2 + 0.8, bus_x], [end[1], end[1]], color=NAVY, lw=0.8,
                solid_capstyle="butt", zorder=1.5)
    ax.plot([bus_x, bus_x], [rows[0], rows[-1]], color=NAVY, lw=0.8, solid_capstyle="projecting",
            zorder=1.5)
    arrow_xy(ax, (bus_x, rows[-1]), (bus_x, bot_y + bot_h / 2), shrink_a=0.0)

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
    h_px, w_px = plt.imread(OUT).shape[:2]
    got = w_px / DPI
    if abs(got - include_width_in()) > 0.01 * include_width_in():
        raise SystemExit(f"{OUT.name} is {got:.3f} in wide; the .tex includes it at {include_width_in():.3f} in")
    print(f"wrote {OUT} ({got:.3f} x {h_px / DPI:.3f} in; included at {include_width_in():.3f} in wide)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
