"""text.py — figures for the ``lookahead/text`` family (embedding + judge-free text evals).

- :func:`repertoire_occupancy_fig` — per arm, the share of therapist turns in each base-repertoire
  cluster by iteration (stacked area; the biggest base clusters named, the rest pooled, the
  out-of-repertoire ``novel`` share on top). The learned/unlearned picture.
- :func:`learned_unlearned_fig` — per arm, the clusters that grew / shrank most, final vs base.
- :func:`drift_fig` — displacement from the base centroid by iteration + the cosine between the two
  K arms' displacements (did they learn the same thing?) and the update-direction alignment.
- :func:`diversity_fig` — template similarity, between-persona variance share, near-duplicate
  rate and distinct-2 by iteration, one line per arm.
- :func:`levels_fig` — the per-conversation text metrics (mean ± SE) by iteration.
- :func:`k_text_forest` — persona-paired K=0 − K=5 *dz* on every text metric at one iteration.
- :func:`profile_fig` — therapist features by within-session turn bin at the endpoint (+ base).

Contract as everywhere in ``plotting``: tidy frames in, ``fig`` out, no disk, ``None`` when the
arms are absent. K=0 solid / K=5 dashed via :data:`K_STYLE`.
"""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import arm_label, k_of
from ..plotting_style import arm_palette, grid
from ._shared import K_STYLE

__all__ = ["repertoire_occupancy_fig", "learned_unlearned_fig", "drift_fig", "diversity_fig",
           "levels_fig", "k_text_forest", "profile_fig"]

_ARMS = ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")
_XLAB = "training iteration (policy that generated the conversations)"


def _style(arm: str) -> dict:
    try:
        return K_STYLE.get(k_of(arm), K_STYLE[0])
    except Exception:
        return K_STYLE[0]


def _present(frame: pd.DataFrame, arms: Optional[Sequence[str]]) -> list:
    have = set(frame["arm"].unique())
    return [a for a in (arms or _ARMS) if a in have]


def repertoire_occupancy_fig(state: pd.DataFrame, labels: pd.DataFrame, k: int, *,
                             arms: Optional[Sequence[str]] = None, top: int = 10):
    """Stacked occupancy by iteration, one panel per arm. Clusters ranked by base share; the
    ``top`` named (``c<i> top_words``), the remainder pooled as *other*, ``novel`` on top."""
    arms = _present(state, arms)
    if not arms:
        return None
    lab = labels.set_index("cluster")
    order = lab["share_base"].sort_values(ascending=False).index.tolist()
    named, rest = order[:top], order[top:]
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(named)}
    fig, axes = grid(len(arms), ncols=2, panel=(5.8, 3.6))
    for ax, arm in zip(axes, arms):
        g = state[state["arm"] == arm].sort_values("iteration")
        x = g["iteration"].to_numpy()
        # Cluster shares are shares of ALL turns (novel turns keep their nearest cluster), so
        # rescale to (1 − novel) and stack novel on top: the panel sums to one.
        scale = 1.0 - g["novel_share"].to_numpy()
        series = [g[f"c{c}"].to_numpy() * scale for c in named]
        series.append(sum(g[f"c{c}"].to_numpy() for c in rest) * scale if rest else np.zeros(len(g)))
        series.append(g["novel_share"].to_numpy())
        names = [f"c{c} {lab.loc[c, 'top_words']}" for c in named] + [f"other ({len(rest)} clusters)", "novel (out of repertoire)"]
        cols = [colors[c] for c in named] + ["#cccccc", "#333333"]
        ax.stackplot(x, *series, labels=names, colors=cols, alpha=0.9, lw=0.3, edgecolor="white")
        ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 1)
        ax.set_title(arm_label(arm), fontsize=10)
        ax.set_xlabel(_XLAB, fontsize=8); ax.set_ylabel("share of therapist turns", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    fig.suptitle(f"Base-repertoire occupancy by iteration (k = {k} clusters fit on the base policy)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def learned_unlearned_fig(lu: pd.DataFrame, *, arms: Optional[Sequence[str]] = None):
    """Horizontal bars of Δ share (final − base) for the top learned / unlearned clusters per arm."""
    arms = _present(lu, arms)
    if not arms:
        return None
    fig, axes = grid(len(arms), ncols=2, panel=(6.2, 3.6))
    for ax, arm in zip(axes, arms):
        g = lu[lu["arm"] == arm].sort_values("delta")
        y = np.arange(len(g))
        colors = ["#D55E00" if d < 0 else "#009E73" for d in g["delta"]]
        ax.barh(y, g["delta"], color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"c{int(c)} {w}" for c, w in zip(g["cluster"], g["top_words"])], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        fi = int(g["final_iter"].iloc[0]) if len(g) else "?"
        ax.set_title(f"{arm_label(arm)} — iteration {fi} vs base", fontsize=10)
        ax.set_xlabel("Δ share of therapist turns (final − base)", fontsize=8)
    fig.suptitle("Learned (green) and unlearned (orange) repertoire clusters", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def drift_fig(drift: pd.DataFrame, cosines: pd.DataFrame, align: Optional[pd.DataFrame] = None, *,
              arms: Optional[Sequence[str]] = None, palette: Optional[Dict[str, str]] = None):
    """(a) ‖centroid − base‖ by iteration per arm; (b) cos between the K=0 and K=5 displacement per
    method (+ PTO vs GRPO at each K); (c) cos(update direction, realised drift) if given."""
    arms = _present(drift, arms)
    if not arms:
        return None
    n = 3 if align is not None and not align.empty else 2
    fig, axes = grid(n, ncols=n, panel=(5.2, 3.5))
    pal = palette or arm_palette(arms)
    ax = axes[0]
    for arm in arms:
        d = drift[drift["arm"] == arm].sort_values("iteration")
        ax.plot(d["iteration"], d["drift_norm"], color=pal.get(arm), lw=2, ms=5, label=arm_label(arm), **_style(arm))
    ax.set_xlabel(_XLAB, fontsize=8); ax.set_ylabel("‖centroid − pooled base centroid‖", fontsize=8)
    ax.set_title("How far each policy moved", fontsize=10); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    if not cosines.empty:
        for col, lab, c, ls in (("cos_K0_K5_PTO", "PTO: K=0 vs K=5", "#0072B2", "-"),
                                ("cos_K0_K5_GRPO", "GRPO: K=0 vs K=5", "#D55E00", "-"),
                                ("cos_PTO_GRPO_K0", "K=0: PTO vs GRPO", "#555555", ":"),
                                ("cos_PTO_GRPO_K5", "K=5: PTO vs GRPO", "#999999", ":")):
            if col in cosines.columns and cosines[col].notna().any():
                ax.plot(cosines["iteration"], cosines[col], color=c, ls=ls, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
    ax.set_xlabel(_XLAB, fontsize=8); ax.set_ylabel("cosine of displacement vectors", fontsize=8)
    ax.set_title("Did the two arms move the same way?", fontsize=10); ax.legend(fontsize=7, frameon=False)
    if n == 3:
        ax = axes[2]
        for arm in _present(align, arms):
            d = align[align["arm"] == arm].sort_values("iteration")
            ax.plot(d["iteration"], d["cos_pooled_vs_cumdrift"], color=pal.get(arm), lw=2, ms=5,
                    label=arm_label(arm), **_style(arm))
        ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-1, 1)
        ax.set_xlabel(_XLAB, fontsize=8); ax.set_ylabel("cos(pooled update direction, drift)", fontsize=8)
        ax.set_title("Did the policy move where the update pushed?", fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def diversity_fig(div: pd.DataFrame, *, arms: Optional[Sequence[str]] = None,
                  palette: Optional[Dict[str, str]] = None):
    """Four diversity scalars by iteration, one line per arm."""
    arms = _present(div, arms)
    if not arms:
        return None
    panels = [("template_sim", "template similarity\n(mean pairwise cos across personas at matched turn)"),
              ("persona_var_share", "between-persona variance share\n(higher = tailored to the patient)"),
              ("dup_rate", "near-duplicate rate\n(share of turns with a ≥0.95-cos twin in another conversation)"),
              ("distinct_2", "distinct-2\n(unique / total bigrams, fixed 400-turn sample)")]
    fig, axes = grid(len(panels), ncols=2, panel=(5.6, 3.4))
    pal = palette or arm_palette(arms)
    for ax, (col, ylab) in zip(axes, panels):
        for arm in arms:
            d = div[div["arm"] == arm].sort_values("iteration")
            ax.plot(d["iteration"], d[col], color=pal.get(arm), lw=2, ms=5, label=arm_label(arm), **_style(arm))
            if col == "persona_var_share" and f"{col}_lo" in d.columns:
                ax.fill_between(d["iteration"], d[f"{col}_lo"], d[f"{col}_hi"], color=pal.get(arm), alpha=0.12, lw=0)
        ax.set_xlabel(_XLAB, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Diversity and persona sensitivity of the therapist's turns", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def levels_fig(levels: pd.DataFrame, metrics: Sequence[str], labels: Dict[str, str], *,
               arms: Optional[Sequence[str]] = None, palette: Optional[Dict[str, str]] = None,
               lower_better: Sequence[str] = ()):
    """Per-conversation text metrics (mean ± SE over 96 personas) by iteration, one panel each."""
    arms = _present(levels, arms)
    metrics = [m for m in metrics if m in levels.columns]
    if not arms or not metrics:
        return None
    fig, axes = grid(len(metrics), ncols=3, panel=(5.2, 3.3))
    pal = palette or arm_palette(arms)
    for ax, m in zip(axes, metrics):
        for arm in arms:
            d = levels[levels["arm"] == arm].sort_values("iteration")
            ax.plot(d["iteration"], d[m], color=pal.get(arm), lw=2, ms=5, label=arm_label(arm), **_style(arm))
            if f"{m}_se" in d.columns:
                ax.fill_between(d["iteration"], d[m] - d[f"{m}_se"], d[m] + d[f"{m}_se"],
                                color=pal.get(arm), alpha=0.12, lw=0)
        ttl = m + (" (lower = better)" if m in lower_better else "")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(_XLAB, fontsize=7); ax.set_ylabel(labels.get(m, m), fontsize=7, wrap=True)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Per-conversation text metrics by iteration (mean ± SE, 96 personas)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def k_text_forest(kt: pd.DataFrame, *, iteration: Optional[int] = None,
                  methods: Sequence[str] = ("PTO", "GRPO"), lower_better: Sequence[str] = (),
                  labels: Optional[Dict[str, str]] = None):
    """Persona-paired K=0 − K=5 *dz* on every text metric at one iteration (default: the last one
    each method has), one panel per method. Lower-better metrics are sign-flipped so that a bar to
    the RIGHT always means *K=0 better*; the axis says so."""
    if kt.empty:
        return None
    labels = labels or {}
    fig, axes = grid(len(methods), ncols=len(methods), panel=(5.6, 3.8))
    for ax, method in zip(axes, methods):
        d = kt[kt["method"] == method]
        if d.empty:
            ax.set_visible(False)
            continue
        it = iteration if iteration is not None and iteration in set(d["iteration"]) else int(d["iteration"].max())
        d = d[d["iteration"] == it].copy()
        d["dz_plot"] = np.where(d["metric"].isin(lower_better), -d["dz"], d["dz"])
        d["lo"] = np.where(d["metric"].isin(lower_better), -d["ci_hi"], d["ci_lo"]) / d["mean_delta"].abs().replace(0, np.nan) * d["dz"].abs()
        d["hi"] = np.where(d["metric"].isin(lower_better), -d["ci_lo"], d["ci_hi"]) / d["mean_delta"].abs().replace(0, np.nan) * d["dz"].abs()
        d = d.sort_values("dz_plot")
        y = np.arange(len(d))
        colors = ["#009E73" if v > 0 else "#D55E00" for v in d["dz_plot"]]
        ax.barh(y, d["dz_plot"], color=colors, alpha=0.85)
        err = np.vstack([np.clip(d["dz_plot"] - d["lo"], 0, None), np.clip(d["hi"] - d["dz_plot"], 0, None)])
        ax.errorbar(d["dz_plot"], y, xerr=err, fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
        for yi, (dz, p) in enumerate(zip(d["dz_plot"], d["p_holm"])):
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            if star:
                ax.text(dz + (0.03 if dz >= 0 else -0.03), yi, star, va="center",
                        ha="left" if dz >= 0 else "right", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([labels.get(m, m) for m in d["metric"]], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        n_lo, n_hi = int(d["n"].min()), int(d["n"].max())
        n_txt = f"n = {n_hi}" if n_lo == n_hi else f"n = {n_lo}–{n_hi} persona pairs, varies by metric"
        ax.set_title(f"{method}: K=0 − K=5 at iteration {it} ({n_txt})", fontsize=9)
        ax.set_xlabel("paired dz  (→ K=0 better; lower-better metrics sign-flipped; Holm stars)", fontsize=8)
    fig.tight_layout()
    return fig


def profile_fig(prof: pd.DataFrame, features: Sequence[str], *, arms: Optional[Sequence[str]] = None,
                iteration_by_arm: Optional[Dict[str, int]] = None, palette: Optional[Dict[str, str]] = None,
                bins: Sequence[str] = ("1-2", "3-5", "6-9", "10+")):
    """Therapist features by within-session turn bin: each arm at its endpoint (coloured) and the
    pooled base (grey, dotted). One panel per feature."""
    arms = _present(prof, arms)
    features = [f for f in features if f in prof.columns]
    if not arms or not features:
        return None
    fig, axes = grid(len(features), ncols=3, panel=(5.0, 3.3))
    pal = palette or arm_palette(arms)
    x = np.arange(len(bins))
    base = prof[prof["iteration"] == 0]
    for ax, f in zip(axes, features):
        if not base.empty:
            b = base.groupby("bin")[f].mean().reindex(bins)
            ax.plot(x, b.values, color="#555555", ls=":", lw=2, marker="d", ms=5, label="base (pooled)")
        for arm in arms:
            g = prof[prof["arm"] == arm]
            it = (iteration_by_arm or {}).get(arm, int(g["iteration"].max()))
            d = g[g["iteration"] == it].set_index("bin")[f].reindex(bins)
            ax.plot(x, d.values, color=pal.get(arm), lw=2, ms=5, label=f"{arm_label(arm)} @ it {it}", **_style(arm))
        ax.set_xticks(x); ax.set_xticklabels(list(bins))
        ax.set_xlabel("therapist turn index within the session", fontsize=8)
        ax.set_ylabel(f, fontsize=8); ax.set_title(f, fontsize=9)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Within-session profile of the therapist's turns (endpoint vs base)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
