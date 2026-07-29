r"""Hand-authored METHOD schematics for Exp3 — the diagrams that explain how PTO and GRPO work.

Four figures, deliberately in two matched pairs:

    pto_framework.png        the iteration loop            }  the ICLR-2025 paper's Figure 1,
    grpo_framework.png       the same loop for GRPO        }  redrawn for Exp3 + its GRPO twin

    pto_preference_tree.png  inside ONE branch point       }  the paper's Figure 2, redrawn for
    grpo_group_rollout.png   inside ONE group              }  greedy mode + its GRPO twin

The two generation figures share a row grid on purpose — put them side by side and the rows line
up, so the only visible differences are the real ones: what the oracle's scores are USED for, how
many candidates survive, and whether the winner feeds back into the next branch point.

These are schematics, not results: nothing here reads the data, so they do NOT live under
``eda/results/`` (which is regenerated per view and per judge by ``tools/render_views.py``) and
they carry no ``<judge>/`` level. They are consumed by the meeting decks and are meant to be
reused in the thesis.

Run:  & ..\..\.venv\Scripts\python.exe build_method_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette — the arm colours come from eda_analysis.plotting_style so a schematic and a results
# figure never disagree about which colour means PTO.
PTO_B  = "#0072B2"   # _ARM_COLORS["PTO_LA0"]
GRPO_O = "#D55E00"   # _ARM_COLORS["GRPO_LA0"]
NAVY   = "#1F3A5F"
GREEN  = "#008A63"
RED    = "#C04A1A"
GREY   = "#5A5A5A"
EDGE   = "#8A96A3"

# (fill, edge) per box role. Roles mirror the paper's colour coding: green = data we produce,
# purple = an API model we call, blue = the policy being trained, orange = the oracle.
ROLE = {
    "source":  ("#EAF1E7", "#5B8C5A"),
    "api":     ("#EDE6F3", "#7B5EA7"),
    "policy":  ("#DCEBF5", PTO_B),
    "policy2": ("#DCEBF5", GRPO_O),
    "data":    ("#DCEDE3", GREEN),
    "oracle":  ("#FBE8D5", GRPO_O),
    "neutral": ("#EFF2F6", EDGE),
    "chosen":  ("#DCEDE3", GREEN),
    "reject":  ("#F7E3DC", RED),
    "update":  ("#FFF4E0", "#B8860B"),
}


def node(ax, x, y, w, h, text, role="neutral", fs=8.5, bold=False, pad=0.28, lw=1.2):
    """Rounded box centred on (x, y). Returns the box so arrows can anchor to its edges."""
    fill, edge = ROLE[role]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={pad},rounding_size={min(w, h) * 0.18:.3f}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#22282F", zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, side="auto", color=NAVY, lw=1.15, style="-|>", rad=0.0, ls="-"):
    """Arrow from box a to box b, anchored on facing edges."""
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "auto":
        side = "h" if abs(bx0 - ax0) >= abs(by0 - ay0) else "v"
    if side == "h":
        sx = ax0 + aw / 2 * (1 if bx0 > ax0 else -1); sy = ay0
        ex = bx0 - bw / 2 * (1 if bx0 > ax0 else -1); ey = by0
    else:
        sx = ax0; sy = ay0 + ah / 2 * (1 if by0 > ay0 else -1)
        ex = bx0; ey = by0 - bh / 2 * (1 if by0 > ay0 else -1)
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (ex, ey), arrowstyle=style, mutation_scale=9,
        color=color, linewidth=lw, linestyle=ls, zorder=1.5,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def elbow(ax, pts, color=NAVY, lw=1.15, ls="-"):
    """Poly-line with an arrowhead on the last leg — for the feedback loops."""
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(
            pts[i], pts[i + 1], arrowstyle="-|>" if last else "-",
            mutation_scale=9 if last else 0, color=color, linewidth=lw, linestyle=ls,
            zorder=1.5, shrinkA=0, shrinkB=1.5 if last else 0))


def label(ax, x, y, text, fs=7.5, color=GREY, ha="center", va="center", bold=False, style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=4,
            fontweight="bold" if bold else "normal", fontstyle=style, linespacing=1.35)


def bracket(ax, x, y0, y1, text, color=GREY, fs=7.5, tick=1.1):
    """Left-hand span bracket with a rotated label — the paper's 'K look-ahead steps' device."""
    ax.plot([x, x], [y0, y1], color=color, lw=1.0, zorder=1)
    ax.plot([x, x + tick], [y0, y0], color=color, lw=1.0, zorder=1)
    ax.plot([x, x + tick], [y1, y1], color=color, lw=1.0, zorder=1)
    ax.text(x - tick * 0.9, (y0 + y1) / 2, text, rotation=90, ha="center", va="center",
            fontsize=fs, color=color, zorder=4)


def canvas(w_in, h_in, xlim, ylim, legend=None, note=None):
    """A bare drawing surface — no in-figure title.

    The title lives in the slide's title bar (decks) or the LaTeX caption (thesis); repeating it
    inside the PNG only duplicates it wherever the figure is used. What DOES belong inside is the
    colour legend, so the figure stays readable on its own — it goes at the bottom, with an
    optional one-line note under it.
    """
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("auto"); ax.axis("off")
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.035
    if note:
        ax.text(xlim[0] + 1, y, note, ha="left", va="bottom", fontsize=7.5,
                color=NAVY, fontstyle="italic")
        y += (ylim[1] - ylim[0]) * 0.055
    if legend:
        ax.text(xlim[0] + 1, y, legend, ha="left", va="bottom", fontsize=7.5, color=GREY)
    return fig, ax


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# =====================================================================
# 1 · PTO framework — the iteration loop  (redraw of ICLR 2025 Figure 1)
# =====================================================================
def pto_framework():
    fig, ax = canvas(11.4, 4.15, (0, 118), (-12, 47),
                     legend="Green = data this iteration produces · purple = an API model we call "
                            "· blue = the policy being trained · orange = the oracle",
                     note="The 96 conversations generated at the start of the iteration are also "
                          "the evaluation set for π$_n$ — there is no separate evaluation pass.")

    src   = node(ax, 11, 26, 17, 11, "96 patient\nsystem prompts\n(one per persona)", "source", 8)
    orc   = node(ax, 40, 41, 21, 8.5, "Oracle\ngpt-4o-mini · Q1+Q2", "oracle", 8)
    pat   = node(ax, 40, 26, 21, 8.5, "Patient model\ngpt-4o-mini", "api", 8)
    pol   = node(ax, 40, 11, 21, 8.5, "Policy  π$_n$\nLlama-3.2-1B + LoRA", "policy", 8)
    tree  = node(ax, 72, 26, 20, 21,
                 "Preference tree data\n\n(prompt, chosen, rejected)\n\nkept only where\n"
                 "Δscore > τ", "data", 8)
    nxt   = node(ax, 104, 26, 19, 8.5, "Policy  π$_{n+1}$", "policy", 8.5, bold=True)

    arrow(ax, src, pat)
    for b in (orc, pat, pol):
        arrow(ax, b, tree, side="h")
    arrow(ax, tree, nxt)
    label(ax, 88, 29.2, "DPO update", fs=8, color=NAVY, bold=True)

    # feedback loop: the new policy becomes next iteration's policy
    elbow(ax, [(104, 21.4), (104, 3.0), (40, 3.0), (40, 6.4)])
    label(ax, 72, 4.6, "next iteration:  π$_{n+1}$ → π$_n$", fs=7.5)

    save(fig, "pto_framework.png")


# =====================================================================
# 2 · GRPO framework — the same loop, one method over
# =====================================================================
def grpo_framework():
    fig, ax = canvas(11.4, 4.15, (0, 118), (-12, 47),
                     legend="Same colour code as the PTO framework figure. The one structural "
                            "difference is where the oracle sits: inside the update, not before it.",
                     note="As in PTO the rollout conversations double as the evaluation set for "
                          "π$_n$ — but the prompt list is fixed for the iteration, every prompt "
                          "trains, and nothing is discarded.")

    src  = node(ax, 11, 22, 17, 11, "96 patient\nsystem prompts\n(one per persona)", "source", 8)
    pat  = node(ax, 37, 32, 19, 8.5, "Patient model\ngpt-4o-mini", "api", 8)
    pol  = node(ax, 37, 12, 19, 8.5, "Policy  π$_n$\nLlama-3.2-1B + LoRA", "policy2", 8)
    roll = node(ax, 65, 22, 20, 15,
                "96 rollout\nconversations\n\nsliced after every\npatient turn with\n"
                "≥ MCL utterances", "data", 8)
    upd  = node(ax, 93, 21, 20, 19,
                "GRPO update\n\nG = 8 completions\nper prompt\n\nA$_g$ = (r$_g$ − mean) / std\n"
                "+ β·KL to π$_{ref}$", "update", 8)
    orc  = node(ax, 93, 42, 22, 8, "Oracle\ngpt-4o-mini · Q1+Q2", "oracle", 8)
    nxt  = node(ax, 93, 3.2, 20, 5.6, "Policy  π$_{n+1}$", "policy2", 8.5, bold=True)

    arrow(ax, src, pat)
    arrow(ax, pat, roll, side="h")
    arrow(ax, pol, roll, side="h")
    arrow(ax, roll, upd)
    arrow(ax, orc, upd, side="v")
    arrow(ax, upd, nxt, side="v")
    label(ax, 106.5, 33.5, "reward for\nEVERY one of\nthe G completions", fs=7, color=GRPO_O)

    elbow(ax, [(83, 3.2), (26, 3.2), (26, 7.75)])
    label(ax, 55, 4.6, "next iteration:  π$_{n+1}$ → π$_n$", fs=7.5)

    save(fig, "grpo_framework.png")


# ---------------------------------------------------------------------
# The two generation figures below share this row grid, so they line up
# when placed side by side.
# ---------------------------------------------------------------------
ROWS = dict(root=118, cand=105, la1=94, la2=84.5, dots=76.5, traj=67,
            oracle=55, score=44, agg=33, keep=21, reply=11.5, next=2.0)
COLS = (18, 50, 82)


def _generation_top(ax, root_text, cand_prefix, n_label, role_cand):
    """Rows shared by both generation figures: root → candidates → look-ahead → trajectories."""
    R = ROWS
    root = node(ax, 50, R["root"], 62, 8, root_text, "api", 8.5)
    bracket(ax, 3.5, R["la1"] + 4.2, R["dots"] - 3.0, "K = 5 look-ahead utterances", fs=7.5)
    label(ax, 8.5, R["cand"] + 7.0, n_label, fs=7.5, color=NAVY, bold=True)

    cands, trajs = [], []
    for i, x in enumerate(COLS):
        tag = ("1", "2", "M" if role_cand == "chosen" else "G")[i]
        c = node(ax, x, R["cand"], 26, 7.2, f"{cand_prefix}$_{{i,{tag}}}$", "data", 8)
        node(ax, x, R["la1"], 26, 6.2, "patient reply", "api", 7.5)
        node(ax, x, R["la2"], 26, 6.2, "therapist reply", "policy", 7.5)
        label(ax, x, R["dots"], "⋮", fs=11, color=GREY)
        t = node(ax, x, R["traj"], 26, 7.2, f"trajectory$_{{i,{tag}}}$", "api", 8)
        arrow(ax, root, c, side="v")
        cands.append(c); trajs.append(t)
        for a, b in ((R["cand"], R["la1"]), (R["la1"], R["la2"])):
            ax.add_patch(FancyArrowPatch((x, a - 3.9), (x, b + 3.4), arrowstyle="-|>",
                                         mutation_scale=8, color=NAVY, lw=1.05, zorder=1.5))
        ax.add_patch(FancyArrowPatch((x, R["dots"] - 2.2), (x, R["traj"] + 3.9), arrowstyle="-|>",
                                     mutation_scale=8, color=NAVY, lw=1.05, zorder=1.5))

    orc = node(ax, 50, R["oracle"], 74, 7.6,
               "Oracle  —  gpt-4o-mini scores the FULL trajectory on Q1+Q2", "oracle", 8.5)
    for t in trajs:
        arrow(ax, t, orc, side="v")
    return root, cands, orc


def pto_preference_tree():
    R = ROWS
    fig, ax = canvas(9.6, 11.0, (-4, 120), (-24, 126),
                     legend="M candidates are generated, all M are scored, and exactly two of "
                            "them survive into the training set.")

    _, _, orc = _generation_top(
        ax, "Trunk$_i$  —  conversation so far  (≥ MCL = 12 utterances)",
        "candidate", "M = 8\ncandidates", "chosen")

    scores = []
    for i, x in enumerate(COLS):
        tag = ("1", "2", "M")[i]
        sc = node(ax, x, R["score"], 24, 6.4, f"score$_{{i,{tag}}}$", "neutral", 8)
        arrow(ax, orc, sc, side="v")
        scores.append(sc)

    agg = node(ax, 50, R["agg"], 74, 7.4,
               "best and worst  ·  emit a pair ONLY if  score$_{best}$ − score$_{worst}$ > τ",
               "neutral", 8.5)
    for sc in scores:
        arrow(ax, sc, agg, side="v")

    rej = node(ax, 22, R["keep"], 32, 7.2, "rejected = worst", "reject", 8, bold=True)
    cho = node(ax, 70, R["keep"], 32, 7.2, "chosen = best", "chosen", 8, bold=True)
    arrow(ax, agg, rej, side="v"); arrow(ax, agg, cho, side="v")

    # the DPO training pair leaves the tree — the ONLY thing this branch point contributes
    pair = node(ax, 24, R["reply"] - 0.5, 46, 7.0,
                "training pair\n(trunk$_i$, chosen, rejected)  →  DPO", "data", 7.8, bold=True)
    arrow(ax, rej, pair, side="v")
    arrow(ax, cho, pair, side="h")

    rep = node(ax, 74, R["reply"], 28, 6.4, "patient reply", "api", 7.5)
    arrow(ax, cho, rep, side="v")
    nxt = node(ax, 60, R["next"], 62, 7.2,
               "Trunk$_{i+1}$  =  trunk$_i$  +  chosen  +  patient reply", "api", 8.5)
    arrow(ax, rep, nxt, side="v")

    for x in COLS:
        label(ax, x, -7.5, "⋮", fs=11, color=GREY)
        ax.add_patch(FancyArrowPatch((60, -1.7), (x, -5.6), arrowstyle="-|>", mutation_scale=8,
                                     color=NAVY, lw=1.05, zorder=1.5))
    label(ax, 50, -12.5, "branch again from the new trunk, until it reaches the target length",
          fs=7.5, color=GREY, style="italic")

    label(ax, 100, R["keep"] + 3.0,
          "greedy:\nthe winner is\nappended to the\ntrunk, so this\nchoice conditions\n"
          "the NEXT\nbranch point",
          fs=7.5, color=PTO_B, ha="left", va="top", bold=True)
    save(fig, "pto_preference_tree.png")


def grpo_group_rollout():
    R = ROWS
    fig, ax = canvas(9.6, 11.0, (-4, 120), (-24, 126),
                     legend="G completions are generated, all G are scored, and all G carry a "
                            "gradient. Rows match the PTO figure so the two can be compared.")

    _, _, orc = _generation_top(
        ax, "Prompt$_i$  —  conversation prefix sliced from the rollout  (≥ MCL = 12)",
        "completion", "G = 8\ncompletions", "group")

    rewards = []
    for i, x in enumerate(COLS):
        tag = ("1", "2", "G")[i]
        rw = node(ax, x, R["score"], 24, 6.4, f"reward  r$_{tag}$", "neutral", 8)
        arrow(ax, orc, rw, side="v")
        rewards.append(rw)

    agg = node(ax, 50, R["agg"], 78, 7.4,
               "group-relative advantage   A$_g$ = ( r$_g$ − mean$_g$ r ) / std$_g$ r",
               "update", 8.5)
    for rw in rewards:
        arrow(ax, rw, agg, side="v")

    advs = []
    for i, x in enumerate(COLS):
        tag = ("1", "2", "G")[i]
        a = node(ax, x, R["keep"], 24, 7.2, f"A$_{tag}$ · completion$_{tag}$", "chosen", 8)
        arrow(ax, agg, a, side="v")
        advs.append(a)

    step = node(ax, 50, R["next"] + 2.5, 82, 8.4,
                "PPO-clipped policy-gradient step   +   β · KL( π ‖ π$_{ref}$ )", "update",
                8.5, bold=True)
    for a in advs:
        arrow(ax, a, step, side="v")

    label(ax, 100, R["keep"] + 3.0,
          "no trunk:\nevery completion\ncontributes a\ngradient, and\nthe prompt list\n"
          "is FIXED for\nthe iteration",
          fs=7.5, color=GRPO_O, ha="left", va="top", bold=True)
    label(ax, 50, -12.5,
          "The next prompt is the next slice on the list — it is not conditioned on which "
          "completion won here.", fs=7.5, color=GREY, style="italic")
    save(fig, "grpo_group_rollout.png")


if __name__ == "__main__":
    pto_framework()
    grpo_framework()
    pto_preference_tree()
    grpo_group_rollout()
