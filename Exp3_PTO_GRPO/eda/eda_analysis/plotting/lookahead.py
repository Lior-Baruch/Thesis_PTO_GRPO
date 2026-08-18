"""lookahead.py — RQ-i figures: what the K knob changes, beyond the reward it does not move.

The K=0 vs K=5 contrast had exactly one figure family before this module (``k_trajectory_Q1Q2``,
the reward curve), and the reward curve is the one place the two arms agree. These figures read the
BEHAVIOUR channels instead, where they do not:

- :func:`k_channel_trajectory` — a channel's per-iteration rate, both K arms of a method side by
  side (+ any context arms). The onset figure: flat-together, then divergent.
- :func:`k_mechanism_panel` — the same channel at all three levels it has to pass through to
  become behaviour — what the REWARD selects for, what the POLICY generates, what the EVAL
  measures — with K=0 and K=5 overlaid at each. The causal-chain figure.
- :func:`k_channel_forest` — every channel's persona-paired *dz* at one matched iteration, so the
  trade-off (one channel down, another up) is visible in a single frame rather than asserted.

Contract as everywhere in ``plotting``: takes already-built tidy frames, never touches disk,
returns a ``fig`` (the notebook owns ``save_fig``), returns ``None`` when the arms are absent.

⚠ **Indexing.** The eval frames are indexed by ``iteration`` = ``model_iter_n`` (the policy that
generated the conversations). The training-side frames (``pool_mean_by_iter`` /
``weighted_lexical_contrast``) are indexed by ``train_iter n``, which samples from the ITER-START
policy — i.e. the same policy the eval calls ``model_iter_{n-1}``. :func:`k_mechanism_panel` shifts
the training rows by −1 so all three rows of the figure are on one policy axis; nothing else here
mixes the two.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import LOWER_IS_BETTER, arm_label, display_label
from ..plotting_style import arm_palette, grid

# Solid = K=0, dashed = K=5, so the K contrast survives greyscale printing and the arm palette
# still carries the method (PTO cool / GRPO warm). Marker differs too — greyscale + colourblind.
_K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}


def _k_of(arm: str) -> int:
    return 5 if arm.endswith("LA5") else 0


def _style(arm: str) -> dict:
    return _K_STYLE.get(_k_of(arm), _K_STYLE[0])


def _direction_note(channel: str) -> str:
    return " (higher = worse)" if channel in LOWER_IS_BETTER else ""


def _unit(channel: str) -> str:
    """The channel's denominator, for the y-label.

    ⚠ Not cosmetic. The same behaviour appears here as a per-turn RATE and as a raw per-session
    COUNT, and the two answer different questions — a rate can move with the count fixed when the
    arms differ in turn count, which these do. A figure that labels a count "per therapist turn"
    invites exactly the misreading the count was added to prevent.
    """
    if channel.endswith("_rate") or channel.endswith("_per_turn") or channel == "q_per_turn":
        return " — per therapist turn"
    if channel in {"MICI_Severity"}:
        return " — 1-5 global"
    if channel in {"RtoQ", "%CR", "%MICO"}:
        return " — ratio"
    if channel in {"mean_turn_len", "conv_len", "n_th_turns"}:
        return ""
    return " — per session"


def k_channel_trajectory(channel_by_iter: pd.DataFrame, channel: str, *,
                         arms: Sequence[str], ax=None, palette=None,
                         annotate_from: Optional[int] = None, title: Optional[str] = None):
    """One behaviour channel's per-iteration mean, one line per arm (K=0 solid, K=5 dashed).

    ``channel_by_iter`` is any per-(arm, iteration) frame holding ``channel`` as a column —
    :func:`~eda_analysis.behavior.mici_behavior_by_iter`,
    :func:`~eda_analysis.behavior.miti_detail_by_iter` and
    :func:`~eda_analysis.behavior.session_shape_by_iter` all qualify. ``annotate_from`` shades the
    iterations at and after a divergence onset, which is where the paired test starts to bite —
    pass the onset you actually measured, never an eyeballed one.
    """
    arms = [a for a in arms if a in set(channel_by_iter["arm"].unique())]
    if not arms or channel not in channel_by_iter.columns:
        return None
    own_fig = ax is None
    fig, ax = (plt.subplots(figsize=(5.6, 3.6)) if own_fig else (ax.get_figure(), ax))
    pal = palette or arm_palette(arms)
    for arm in arms:
        d = (channel_by_iter[channel_by_iter["arm"] == arm]
             .groupby("iteration")[channel].mean().sort_index())
        ax.plot(d.index, d.values, color=pal.get(arm), lw=2.0, ms=5,
                label=arm_label(arm), **_style(arm))
    if annotate_from is not None:
        ax.axvspan(annotate_from - 0.5, ax.get_xlim()[1], color="#D55E00", alpha=0.06, zorder=0)
        ax.text(annotate_from - 0.4, ax.get_ylim()[1], " divergence onset ",
                fontsize=6.5, va="top", color="#8a3d00", style="italic")
    ax.set_xlabel("training iteration (policy that generated the conversations)")
    ax.set_ylabel(f"{display_label(channel)}{_unit(channel)}", fontsize=8)
    ax.set_title(title or f"{display_label(channel)}{_direction_note(channel)}", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    if own_fig:
        fig.tight_layout()
    return fig


def k_channel_trajectory_grid(channel_by_iter, channels: Sequence[str], *,
                              arms: Sequence[str], ncols: int = 2,
                              suptitle: Optional[str] = None, onset: Optional[dict] = None):
    """:func:`k_channel_trajectory` over several channels in one grid.

    ``channel_by_iter`` may be a single frame or a ``{channel: frame}`` mapping when the channels
    come from different builders (MICI rates and session shape live in different frames).
    ``onset`` is an optional ``{channel: iteration}`` map forwarded as ``annotate_from``.
    """
    def _frame(c):
        return channel_by_iter[c] if isinstance(channel_by_iter, dict) else channel_by_iter

    channels = [c for c in channels if c in getattr(_frame(c), "columns", [])]
    if not channels:
        return None
    fig, axes = grid(len(channels), ncols=ncols, panel=(5.6, 3.4))
    pal = arm_palette(list(arms))
    for i, (ax, c) in enumerate(zip(axes, channels)):
        k_channel_trajectory(_frame(c), c, arms=arms, ax=ax, palette=pal,
                             annotate_from=(onset or {}).get(c))
        # Only the bottom row keeps the (long, shared) x-label; otherwise it repeats four times.
        if i < len(channels) - ncols:
            ax.set_xlabel("")
        ax.get_legend().remove() if ax.get_legend() else None
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels), frameon=False, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, y=1.07, fontweight="bold")
    fig.tight_layout()
    return fig


def k_mechanism_panel(select: pd.DataFrame, generate: pd.DataFrame, evaluate: pd.DataFrame, *,
                      select_col: str, generate_col: str, evaluate_col: str,
                      arms: Sequence[str], suptitle: Optional[str] = None,
                      row_titles: Sequence[str] = ("what the REWARD selects for",
                                                   "what the POLICY generates",
                                                   "what the EVAL measures")):
    """The three-level causal chain for one behaviour, K=0 vs K=5 overlaid at every level.

    A behaviour can only become policy by passing selection → generation → the eval, and a claim
    that a knob "prevents" the behaviour is only mechanistic if the knob shows up at the FIRST
    level. Three stacked axes, shared x (policy iteration):

    1. ``select``   — per-(arm, train_iter) selection weight, e.g. ``pref.weighted_lexical_contrast``
       (``w_overpraise``). **Shifted by −1** onto the policy axis (see the module note).
    2. ``generate`` — per-(arm, train_iter) candidate-pool mean, e.g. ``pref.pool_mean_by_iter``
       (``pool_overpraise``). Same −1 shift.
    3. ``evaluate`` — per-(arm, iteration) eval rate, already on the policy axis.

    Row 1 carries ±1 SE whiskers when a ``<col>_se`` column is present. Returns ``None`` if any
    level is missing its column.
    """
    levels = [(select, select_col, True), (generate, generate_col, True),
              (evaluate, evaluate_col, False)]
    if any(df is None or df.empty or col not in df.columns for df, col, _ in levels):
        return None
    arms = [a for a in arms
            if any(a in set(df["arm"].unique()) for df, _, _ in levels if "arm" in df.columns)]
    if not arms:
        return None
    pal = arm_palette(list(arms))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.2), sharex=True)
    for ax, (df, col, is_train), rtitle in zip(axes, levels, row_titles):
        it_col = "train_iter" if is_train else "iteration"
        if it_col not in df.columns:
            it_col = "iteration" if "iteration" in df.columns else "train_iter"
        for arm in arms:
            d = df[df["arm"] == arm]
            if d.empty:
                continue
            d = d.groupby(it_col)[[col] + ([f"{col}_se"] if f"{col}_se" in d.columns else [])].mean()
            x = d.index.to_numpy(float) - (1.0 if is_train else 0.0)
            ax.plot(x, d[col].to_numpy(), color=pal.get(arm), lw=2.0, ms=5,
                    label=arm_label(arm), **_style(arm))
            if f"{col}_se" in d.columns:
                se = d[f"{col}_se"].to_numpy()
                ax.fill_between(x, d[col] - se, d[col] + se, color=pal.get(arm), alpha=0.15, lw=0)
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.set_title(rtitle, fontsize=10)
        ax.set_ylabel(display_label(col))
    axes[-1].set_xlabel("policy iteration (training rows shifted −1 onto the policy axis)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=len(labels), frameon=False, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def k_channel_forest(k_paired: pd.DataFrame, *, iteration: int, method: str = "PTO",
                     channels: Optional[Sequence[str]] = None, alpha: float = 0.05,
                     title: Optional[str] = None, caption: Optional[str] = None):
    """Every behaviour channel's persona-paired *dz* at ONE matched iteration — the trade-off frame.

    Takes :func:`~eda_analysis.stats.paired_k_comparison` run on
    :func:`~eda_analysis.behavior.channel_scores_long`. Sign convention is that function's:
    ``+ => K=0 higher``. Bars are sorted by *dz*, so "look-ahead removed X but added Y" is one
    picture instead of two sentences. Channels reaching ``p_holm < alpha`` are solid; the rest are
    drawn hollow, because an honest trade-off figure has to show what did NOT move.

    Colour encodes VALENCE, not direction: a channel in ``LOWER_IS_BETTER`` with a positive dz means
    K=0 does more of a bad thing (red = look-ahead helped here); a neutral channel (turn length,
    conversation length) is grey, since neither direction is an improvement.
    """
    if k_paired is None or k_paired.empty:
        return None
    d = k_paired[k_paired["iteration"] == iteration]
    if "method" in d.columns:
        d = d[d["method"] == method]
    if channels is not None:
        d = d[d["metric"].isin(channels)]
    d = d.dropna(subset=["dz"]).sort_values("dz")
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.4, max(3.4, 0.32 * len(d))))
    for i, (_, r) in enumerate(d.iterrows()):
        sig = pd.notna(r.get("p_holm")) and r["p_holm"] < alpha
        if r["metric"] in LOWER_IS_BETTER:
            color = "#D55E00" if r["dz"] > 0 else "#2ca02c"
        else:
            color = "#777777"
        ax.barh(i, r["dz"], color=color if sig else "none", edgecolor=color,
                lw=1.4, height=0.66, zorder=2)
        ax.text(r["dz"] + (0.02 if r["dz"] >= 0 else -0.02), i,
                f"{r['dz']:+.2f}" + ("*" if sig else ""), va="center",
                ha="left" if r["dz"] >= 0 else "right", fontsize=6.5, color="#333333")
    ax.axvline(0, color="#555555", lw=1.0, ls="--")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([display_label(m) for m in d["metric"]], fontsize=7)
    ax.set_xlabel("persona-paired dz,  K=0 - K=5   (+ means the K=0 policy does MORE of it)")
    ax.set_title(title or f"{method}: what look-ahead changed, iteration {iteration} (n=96 personas)")
    pad = 0.18 * max(abs(d["dz"].min()), abs(d["dz"].max()), 0.1)
    ax.set_xlim(d["dz"].min() - pad * 2, d["dz"].max() + pad * 2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#D55E00", label="MI-inconsistent: K=0 does more (look-ahead helped)"),
                       Patch(fc="#2ca02c", label="MI-inconsistent: K=5 does more"),
                       Patch(fc="#777777", label="no valence (session shape)"),
                       Patch(fc="none", ec="#555555", label=f"hollow = not significant (Holm ≥ {alpha})")],
              fontsize=6.5, loc="lower right", frameon=True, framealpha=0.9)
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.5,
                 style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig


def k_cost_benefit(k_paired: pd.DataFrame, *, reward_metric: str = "Q1Q2",
                   hack_channel: str = "MICI_OverPraise_rate",
                   control_channel: Optional[str] = "MICI_BehaviorTotal",
                   method: str = "PTO", title: Optional[str] = None):
    """Reward *dz*, hack-channel *dz*, and the AGGREGATE the channel belongs to, per iteration.

    Three lines on one axis — the paired K contrast on the REWARD the run optimised
    (``reward_metric``), on the behaviour channel it was hacked through (``hack_channel``), and on
    the ``control_channel`` that channel is a component of — all as *dz*, all ``+ => K=0 higher``.

    ⚠ The control line is the point, not decoration. A hack channel collapsing looks like a fix
    until you plot the aggregate beside it; if the aggregate does not move, the policy substituted
    one behaviour for another and "mitigation" is the wrong word. Pass ``control_channel=None``
    only when the channel genuinely has no aggregate.

    Needs all contrasts in one frame — concatenate the rubric and channel runs of
    :func:`~eda_analysis.stats.paired_k_comparison` before calling.
    """
    if k_paired is None or k_paired.empty:
        return None
    d = k_paired[k_paired["method"] == method] if "method" in k_paired.columns else k_paired
    series = {reward_metric: ("#0072B2", "o", "-"), hack_channel: ("#D55E00", "s", "--")}
    if control_channel:
        series[control_channel] = ("#555555", "^", "-.")
    present = [m for m in series if m in set(d["metric"])]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in present:
        col, mk, ls = series[m]
        dd = d[d["metric"] == m].sort_values("iteration")
        ax.plot(dd["iteration"], dd["dz"], color=col, marker=mk, ls=ls, lw=2.0, ms=5,
                label=display_label(m))
        sig = dd[dd["p_holm"] < 0.05]
        ax.scatter(sig["iteration"], sig["dz"], s=110, facecolors="none", edgecolors=col, lw=1.4,
                   zorder=3)
    ax.axhline(0, color="#555555", lw=1.0, ls="--")
    for y, lab in ((0.2, "small"), (0.5, "medium"), (0.8, "large")):
        ax.axhline(y, color="#bbbbbb", lw=0.7, ls=":", zorder=0)
        ax.text(ax.get_xlim()[1], y, f" {lab}", fontsize=6, va="center", color="#999999")
    ax.set_xlabel("matched training iteration  (0 = the two base models: the noise floor)")
    ax.set_ylabel("persona-paired dz  (K=0 - K=5)")
    ax.set_title(title or f"{method}: look-ahead closes one channel, not the aggregate")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.text(0.5, -0.03, "Circled markers are Holm-significant (p<.05); + means the K=0 policy is "
             "higher. The reward line near zero says the arms are equivalent by the objective; the "
             "aggregate line near zero says they are equivalent in TOTAL MI-inconsistency; only "
             "the channel line separates them. That is substitution, not mitigation.",
             ha="center", va="top", fontsize=7, style="italic", color="#444444", wrap=True)
    fig.tight_layout()
    return fig
