"""process.py — figures for the ``lookahead/process`` family (utterance-level MI process coding).

- :func:`yield_fig` — P(next patient utterance = change talk | therapist code) per arm at the
  endpoint, base beside it, one panel per grader.
- :func:`ct_trajectory_fig` — change-talk share by patient turn bin, endpoint vs base, per grader.
- :func:`transition_heatmap` — therapist code × next patient code, one panel per arm endpoint.
- :func:`code_mix_fig` — the therapist code mix (stacked shares) by iteration per arm.

Contract as everywhere in ``plotting``: tidy frames in, ``fig`` out, no disk.
"""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import arm_label, k_of
from ..plotting_style import arm_palette, grid
from ._shared import K_STYLE

__all__ = ["yield_fig", "ct_trajectory_fig", "transition_heatmap", "code_mix_fig"]

_ARMS = ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")
_XLAB = "training iteration (policy that generated the conversations)"


def _style(arm: str) -> dict:
    try:
        return K_STYLE.get(k_of(arm), K_STYLE[0])
    except Exception:
        return K_STYLE[0]


def yield_fig(yields_by_judge: Dict[str, pd.DataFrame], codes: Sequence[str], *,
              arms: Optional[Sequence[str]] = None, palette: Optional[Dict[str, str]] = None,
              min_n: int = 20):
    """Grouped bars: P(change talk | therapist code) at each arm's endpoint + pooled base, with
    Wilson 95 % intervals; codes with fewer than ``min_n`` turns are left blank."""
    judges = list(yields_by_judge)
    if not judges:
        return None
    fig, axes = grid(len(judges), ncols=len(judges), panel=(7.2, 3.8))
    for ax, j in zip(axes, judges):
        y = yields_by_judge[j]
        arms_j = [a for a in (arms or _ARMS) if a in set(y["arm"])]
        pal = palette or arm_palette(arms_j)
        series = [("base", y[y["iteration"] == 0].groupby("th_code").agg(p_ct=("p_ct", "mean"), n=("n", "sum"),
                                                                       p_ct_lo=("p_ct_lo", "mean"), p_ct_hi=("p_ct_hi", "mean")), "#555555")]
        for a in arms_j:
            g = y[y["arm"] == a]; it = int(g["iteration"].max())
            series.append((f"{arm_label(a)} @ it {it}", g[g["iteration"] == it].set_index("th_code"), pal.get(a)))
        w = 0.8 / len(series); x = np.arange(len(codes))
        for i, (lab, d, col) in enumerate(series):
            d = d.reindex(codes)
            ok = d["n"].fillna(0) >= min_n
            vals = d["p_ct"].where(ok)
            err = np.vstack([np.clip(vals - d["p_ct_lo"], 0, None).fillna(0), np.clip(d["p_ct_hi"] - vals, 0, None).fillna(0)])
            ax.bar(x + (i - len(series) / 2 + 0.5) * w, vals, w, color=col, alpha=0.9 if i else 0.6, label=lab,
                   yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.5})
        ax.set_xticks(x); ax.set_xticklabels(list(codes), fontsize=8)
        ax.set_ylabel("P(next patient utterance = change talk)", fontsize=8)
        ax.set_title(f"Yield of each therapist behaviour — {j}", fontsize=10)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return fig


def ct_trajectory_fig(traj_by_judge: Dict[str, pd.DataFrame], *, arms: Optional[Sequence[str]] = None,
                      palette: Optional[Dict[str, str]] = None, bins: Sequence[str] = ("1-2", "3-5", "6-9", "10+")):
    """Change-talk share by patient turn bin: endpoint per arm (coloured) vs pooled base (grey)."""
    judges = list(traj_by_judge)
    if not judges:
        return None
    fig, axes = grid(len(judges), ncols=len(judges), panel=(5.6, 3.6))
    x = np.arange(len(bins))
    for ax, j in zip(axes, judges):
        t = traj_by_judge[j]
        arms_j = [a for a in (arms or _ARMS) if a in set(t["arm"])]
        pal = palette or arm_palette(arms_j)
        b = t[t["iteration"] == 0].groupby("bin")["ct_prop"].mean().reindex(bins)
        ax.plot(x, b.values, color="#555555", ls=":", lw=2, marker="d", label="base (pooled)")
        for a in arms_j:
            g = t[t["arm"] == a]; it = int(g["iteration"].max())
            d = g[g["iteration"] == it].set_index("bin")["ct_prop"].reindex(bins)
            ax.plot(x, d.values, color=pal.get(a), lw=2, ms=5, label=f"{arm_label(a)} @ it {it}", **_style(a))
        ax.set_xticks(x); ax.set_xticklabels(list(bins)); ax.set_ylim(0, 1)
        ax.set_xlabel("patient turn index within the session", fontsize=8)
        ax.set_ylabel("change-talk share of patient utterances", fontsize=8)
        ax.set_title(f"Within-session change talk — {j}", fontsize=10); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return fig


def transition_heatmap(mats: Dict[str, pd.DataFrame], *, title: str = ""):
    """One heatmap per (label → row-normalised therapist × patient matrix), n per row on the right."""
    labs = list(mats)
    if not labs:
        return None
    fig, axes = grid(len(labs), ncols=min(len(labs), 3), panel=(3.6, 4.2))
    for ax, lab in zip(axes, labs):
        M = mats[lab]
        cols = [c for c in M.columns if c != "n"]
        im = ax.imshow(M[cols].to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
        ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{i} (n={int(n)})" for i, n in zip(M.index, M["n"])], fontsize=7)
        for r in range(len(M)):
            for c in range(len(cols)):
                v = M.iloc[r][cols[c]]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else "black")
        ax.set_title(lab, fontsize=9)
    fig.colorbar(im, ax=axes[:len(labs)], shrink=0.6, label="P(next patient code | therapist code)")
    if title:
        fig.suptitle(title, fontweight="bold", y=1.02)
    return fig


def code_mix_fig(levels: pd.DataFrame, codes: Sequence[str], *, arms: Optional[Sequence[str]] = None):
    """Stacked shares of the therapist codes by iteration, one panel per arm."""
    arms = [a for a in (arms or _ARMS) if a in set(levels["arm"])]
    if not arms:
        return None
    cmap = plt.get_cmap("tab20"); colors = {c: cmap(i) for i, c in enumerate(codes)}
    fig, axes = grid(len(arms), ncols=2, panel=(5.8, 3.4))
    for ax, a in zip(axes, arms):
        g = levels[levels["arm"] == a].sort_values("iteration")
        x = g["iteration"].to_numpy()
        ax.stackplot(x, *[g[f"th_{c}_rate"].fillna(0).to_numpy() for c in codes], labels=list(codes),
                     colors=[colors[c] for c in codes], alpha=0.9, lw=0.3, edgecolor="white")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 1)
        ax.set_title(arm_label(a), fontsize=10); ax.set_xlabel(_XLAB, fontsize=8)
        ax.set_ylabel("share of therapist turns", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.suptitle("Therapist code mix by iteration (MIPROC)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
