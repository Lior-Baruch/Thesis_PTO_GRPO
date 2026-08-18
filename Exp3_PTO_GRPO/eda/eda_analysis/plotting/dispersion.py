"""plotting/dispersion.py — figures for :mod:`eda_analysis.dispersion` (does look-ahead WIDEN or
merely RESCALE the training signal?). Promoted 2026-08-18 from the look-ahead paper's
``analysis/dispersion_by_k.py`` (``figures/dispersion_by_k_fig.png`` + ``_fig_tau.png``).

- :func:`dispersion_fig` — 2×2: within-group SD, best−worst margin, the scale-free margin/SD (with
  the shuffle-null band + iid-normal expectation) and the winner's standardized lead, four arms by
  training iteration. K=0 solid + circle, K=5 dashed + square; the arm palette carries the method.
- :func:`tau_fig` — 1×2: PTO pair yield vs τ at train_iter 1 (same base policy in both arms) and
  pooled, raw and after rescaling by the iteration-1 SD ratio.

Contract as everywhere in ``plotting``: takes the tidy frames the module returns, never touches
disk, returns a ``fig`` (the notebook owns ``save_fig``). The grader is the training oracle by
construction (candidate rewards cannot be re-graded), so these figures are NOT judge-swappable.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ..plotting_style import arm_palette

__all__ = ["K_STYLE", "dispersion_fig", "tau_fig"]

# Solid + circle = K=0, dashed + square = K=5 (survives greyscale; the palette carries the method).
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}
_ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
_GRADER = "training oracle (gpt-4o-mini)"


def _k_of(arm: str) -> int:
    return int(arm.split("_LA")[1])


def dispersion_fig(by_iter: pd.DataFrame, expectation: pd.DataFrame, *,
                   arms: Optional[Sequence[str]] = None, grader: str = _GRADER,
                   figsize=(7.2, 5.6)):
    """SD / margin / margin-over-SD / winner-z by training iteration, four arms (2×2).

    ``by_iter`` = :func:`eda_analysis.dispersion.dispersion_by_iter`; ``expectation`` =
    :func:`eda_analysis.dispersion.iid_expectation` (the ``ddof=0`` row supplies the dotted
    reference lines in panels c and d). Panel c also shades the shuffle-null range across all
    arm-iterations. Legend below the grid."""
    arms = list(arms) if arms is not None else [a for a in _ARMS if a in set(by_iter["arm"])]
    pal = arm_palette(arms)
    E0 = expectation[expectation["sd_estimator"] == "ddof=0"].iloc[0]
    exp_rom, exp_wz = float(E0["ratio_of_means"]), float(E0["winner_z_mean"])
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    panels = [("sd_mean", "within-group SD (oracle points)", "(a) within-group SD of the 8 candidate scores"),
              ("margin_mean", "best − worst margin (oracle points)", "(b) best − worst margin"),
              ("margin_over_sd", "mean margin / mean SD", "(c) margin / SD (scale-free)"),
              ("winner_z", "(best − group mean) / SD", "(d) winner's standardized lead")]
    xlab = r"training iteration $n$ (policy $\pi_n$; $\pi_1$ = base)"
    xmax = int(by_iter["train_iter"].max())
    for ax, (col, ylab, ttl) in zip(axes, panels):
        for arm in arms:
            d = by_iter[by_iter.arm == arm].sort_values("train_iter")
            st = K_STYLE[_k_of(arm)]
            ax.plot(d["train_iter"], d[col], color=pal[arm], ls=st["ls"], marker=st["marker"],
                    lw=1.6, ms=5, label=arm.replace("_LA", " K="))
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"{ttl}\n{grader}", fontsize=9)
        ax.set_xticks(range(1, xmax + 1))
        ax.grid(True, alpha=0.3)
    # (c): iid-normal expectation + the shuffle-null range across all arm-iterations
    nlo, nhi = float(by_iter["null_margin_over_sd"].min()), float(by_iter["null_margin_over_sd"].max())
    axes[2].axhspan(nlo, nhi, color="0.5", alpha=0.15, lw=0,
                    label="shuffle-null range, all arm-iterations (scores permuted across groups)")
    axes[2].axhline(exp_rom, color="0.25", ls=":", lw=1.2,
                    label=f"iid-normal expectation, M=8 ({exp_rom:.2f})")
    axes[2].set_ylim(min(by_iter["margin_over_sd"].min(), nlo) - 0.08,
                     max(by_iter["margin_over_sd"].max(), exp_rom, nhi) + 0.08)
    axes[2].text(xmax + 0.4, exp_rom, f"iid {exp_rom:.2f}", fontsize=7, color="0.25", ha="right", va="bottom")
    # (d): iid-normal expectation for the winner's standardized lead
    axes[3].axhline(exp_wz, color="0.25", ls=":", lw=1.2,
                    label=f"iid-normal expectation, M=8 ({exp_wz:.2f})")
    axes[3].set_ylim(min(by_iter["winner_z"].min(), exp_wz) - 0.08,
                     max(by_iter["winner_z"].max(), exp_wz) + 0.08)
    axes[3].text(xmax + 0.4, exp_wz, f"iid {exp_wz:.2f}", fontsize=7, color="0.25", ha="right", va="bottom")
    for ax in axes[:2]:
        ax.set_ylim(bottom=0)
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    h3, l3 = axes[3].get_legend_handles_labels()
    extra, seen = [], set(l0)
    for h, l in zip(h2 + h3, l2 + l3):
        key = l.split(",")[0]
        if key not in seen:
            lab = l.replace(f" ({exp_rom:.2f})", "").replace(f" ({exp_wz:.2f})", "")
            extra.append((h, lab + (" (value printed in panel)" if l.startswith("iid") else "")))
            seen.add(key)
    fig.tight_layout()
    fig.legend(h0 + [h for h, _ in extra], l0 + [l for _, l in extra], loc="upper center",
               ncol=3, fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    return fig


def tau_fig(tau: pd.DataFrame, r_iter1: Optional[float] = None, *, tau_trainer: Optional[float] = None,
            method: Optional[str] = None, grader: str = _GRADER, figsize=(7.2, 3.1)):
    """PTO pair yield vs τ, raw and rescaled (1×2): (a) train_iter 1 (base policy, same in both
    arms), (b) all iterations pooled. ``tau`` = :func:`eda_analysis.dispersion.tau_sensitivity`;
    ``r_iter1`` (the iteration-1 SD ratio printed in the legend), ``tau_trainer`` and ``method``
    default to the frame's ``.attrs``. Legend below the panels."""
    r_iter1 = float(tau.attrs.get("r_iter1")) if r_iter1 is None else float(r_iter1)
    tau_trainer = float(tau.attrs.get("tau_trainer", 0.10)) if tau_trainer is None else float(tau_trainer)
    method = tau.attrs.get("method", "PTO") if method is None else method
    a0, a5 = f"{method}_LA0", f"{method}_LA5"
    pal = arm_palette([a0, a5])
    taus = sorted(float(t) for t in tau["tau"].unique())
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    n_iter = tau[tau.train_iter != "pooled"]["train_iter"].astype(int)
    pooled_lab = f"(b) all iterations {int(n_iter.min())}–{int(n_iter.max())} pooled" if len(n_iter) else "(b) all iterations pooled"
    for ax, (t, ttl) in zip(axes, [(1, "(a) train_iter 1 (base policy, same in both arms)"), ("pooled", pooled_lab)]):
        d = tau[tau.train_iter == t].sort_values("tau")
        c0, c5 = pal[a0], pal[a5]
        ax.plot(d.tau, d.yield_K0_raw, color=c0, ls="-", marker="o", lw=1.7, ms=5.5, label=f"{method} K=0, raw margins")
        ax.plot(d.tau, d.yield_K5_raw, color=c5, ls="--", marker="s", lw=1.7, ms=5.5, label=f"{method} K=5, raw margins")
        ax.plot(d.tau, d.yield_K0_x_r1, color=c0, ls=":", marker="o", mfc="white", lw=1.5, ms=5.5,
                label=f"{method} K=0, margins × {r_iter1:.2f} (iter-1 SD ratio)")
        ax.plot(d.tau, d.yield_K5_div_r1, color=c5, ls="-.", marker="s", mfc="white", lw=1.5, ms=5.5,
                label=f"{method} K=5, margins ÷ {r_iter1:.2f}")
        ax.axvline(tau_trainer, color="0.4", ls=":", lw=1.0)
        ax.text(tau_trainer + 0.004, 0.06, "τ used\nin training", fontsize=7, color="0.3", va="bottom")
        ax.set_xlabel("τ (oracle points)")
        ax.set_title(f"{ttl}\n{grader}", fontsize=9)
        ax.set_xticks(taus)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("pair yield\n(share of branch points with margin > τ)")
    fig.tight_layout()
    hnd, lab = axes[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="upper center", ncol=2, fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    return fig
