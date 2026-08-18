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

Plus the paper's headline figures, promoted 2026-08-18 from
``papers/2026_lookahead_pto_grpo/analysis/{k_contrast_headline,cross_k_multijudge}.py`` and fed by
:mod:`eda_analysis.lookahead` / :mod:`eda_analysis.transfer` frames:

- :func:`k_headline_fourarm` — 2×2: the four arms' Q1+Q2 level per grader (top) over the paired
  K=0 − K=5 delta strip per grader (bottom), with the ±0.10 oracle-repeatability band.
- :func:`k_delta_grid` — the paired delta strip on every rubric (3×3) for ONE grader.
- :func:`k_channels_grid` — the delta strip on the three oracle-coded channels × two graders,
  plus two judge-invariant text channels.
- :func:`k_contrast_both_judges` — the K contrast under BOTH graders on one axis (rows Q1Q2 /
  MICI, cols PTO / GRPO): primary solid + CI ribbon, held-out dotted + CI bars, Holm stars.
- :func:`k_retention` — gain retention (Δ held-out / Δ primary over the arm's own base) by K.
- :func:`k_did` — the method gap at each K per grader over the K × method DiD.

Every delta figure states the sign convention on its axis: ``+ => K=0 higher`` (K=0 minus K=5;
MICI channels lower-better). Iteration 0 = the two arms' independent base draws. GRPO_LA5 is
right-censored at 5, so its series simply stop — the legends say so.

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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ..constants import DISPLAY_NAMES, LOWER_IS_BETTER, arm_label, display_label
from ..plotting_style import arm_palette, grid

__all__ = [
    "K_STYLE", "k_channel_trajectory", "k_channel_trajectory_grid", "k_mechanism_panel",
    "k_channel_forest", "k_cost_benefit",
    "k_headline_fourarm", "k_delta_grid", "k_channels_grid", "k_contrast_both_judges",
    "k_retention", "k_did",
]

#: Solid + circle = K=0, dashed + square = K=5, so the K contrast survives greyscale printing and
#: the arm palette still carries the method (PTO cool / GRPO warm). Marker differs too — greyscale
#: + colourblind. The one place the K line style is defined (the paper's ``C.K_STYLE``).
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}
_K_STYLE = K_STYLE   # back-compat alias

#: The four Exp3 arms in palette / legend order.
_ARMS = ("PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5")
#: Rubrics on the 1-5 / 1-7 point scale that the ±oracle-noise band applies to (PCT / MICI are 0-1).
_FIVE_POINT = ("Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI")
_RUBRICS = ("Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI")
_LOWER_BETTER = set(LOWER_IS_BETTER) | {"MICI"}


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


# ═════════════════════════════════════════════════════════════════════════════
# The paper's headline figures (promoted 2026-08-18) — fed by eda_analysis.lookahead / .transfer
# ═════════════════════════════════════════════════════════════════════════════

def _oracle_noise(value) -> float:
    """The ±band drawn on point-scale delta panels: pass ``S.ORACLE_NOISE``; ``None`` resolves
    ``EdaConfig().oracle_noise`` lazily (the one owner of that number)."""
    if value is not None:
        return float(value)
    from ..config import EdaConfig
    return float(EdaConfig().oracle_noise)


def _judges(frames: pd.DataFrame, judges) -> list:
    """Judge labels to draw, in column order: as given, else the frame's labels with the PRIMARY
    oracle's label first (frames sorted alphabetically would otherwise put the held-out judge left)."""
    if judges is not None:
        return list(judges)
    labs = list(dict.fromkeys(frames["judge"].tolist())) if len(frames) else []
    from ..constants import judge_dirname, PRIMARY_JUDGE_TAG
    prim = judge_dirname(PRIMARY_JUDGE_TAG)
    return ([prim] if prim in labs else []) + [l for l in labs if l != prim]


def _pal(palette) -> dict:
    return dict(palette) if palette else arm_palette(list(_ARMS))


def _delta_strip(ax, df, method, color, xoff, *, ms=5.5, lw=1.4, capsize=2.2, alpha=0.05):
    """One method's paired delta by iteration: 95 % CI whiskers, a faint connecting line, filled
    markers where Holm ``p < alpha`` and hollow markers otherwise (so n.s. rows stay visible)."""
    d = df[df["method"] == method].sort_values("iteration")
    if d.empty:
        return
    x = d["iteration"].to_numpy(float) + xoff
    y = d["mean_delta"].to_numpy(float)
    yerr = np.vstack([y - d["ci_lo"].to_numpy(float), d["ci_hi"].to_numpy(float) - y])
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=lw, capsize=capsize, zorder=2)
    ax.plot(x, y, ls="-", lw=1.0, color=color, alpha=0.55, zorder=2)
    sig = (d["p_holm"] < alpha).to_numpy()
    ax.plot(x[sig], y[sig], "o", ms=ms, mfc=color, mec=color, zorder=3)
    ax.plot(x[~sig], y[~sig], "o", ms=ms, mfc="white", mec=color, mew=1.4, zorder=3)


def _band(ax, noise: float, *, label=True):
    """Zero line + the ±oracle-repeatability band (grey), optionally labelled."""
    ax.axhline(0, color="#333333", lw=0.9, zorder=1)
    ax.axhspan(-noise, noise, color="#999999", alpha=0.18, lw=0, zorder=0)
    if label:
        ax.text(0.99, noise, f" ±{noise:.2f} oracle repeatability",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.5,
                color="#555555")


def _sig_handles(pal: dict, noise: Optional[float], *, pto_label="PTO", grpo_label="GRPO"):
    h = [Line2D([], [], color=pal["PTO_LA0"], marker="o", ls="-", label=pto_label),
         Line2D([], [], color=pal["GRPO_LA0"], marker="o", ls="-", label=grpo_label),
         Line2D([], [], color="#444444", marker="o", ls="", mfc="#444444", label="Holm p<.05"),
         Line2D([], [], color="#444444", marker="o", ls="", mfc="white", label="n.s.")]
    if noise is not None:
        h.append(plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.25,
                               label=f"±{noise:.2f} oracle repeatability"))
    return h


def _censor_labels(frames: pd.DataFrame):
    """Legend labels that say where each method's matched iterations stop (read off the frame)."""
    def _rng(method):
        d = frames[frames["method"] == method]
        if d.empty:
            return method
        lo, hi = int(d["iteration"].min()), int(d["iteration"].max())
        return f"{method} (iters {lo}–{hi}{', censored' if method == 'GRPO' else ''})"
    return _rng("PTO"), _rng("GRPO")


def k_headline_fourarm(levels_long: pd.DataFrame, frames: pd.DataFrame, *, metric: str = "Q1Q2",
                       judges: Optional[Sequence[str]] = None, palette: Optional[dict] = None,
                       oracle_noise: Optional[float] = None, censor_arm: str = "GRPO_LA5"):
    """THE headline: four-arm level curves (top) over the paired K=0 − K=5 delta (bottom), per grader.

    ``levels_long`` = :func:`eda_analysis.lookahead.k_levels`'s ``"levels_long"``; ``frames`` =
    :func:`eda_analysis.lookahead.paired_k_frames`. One column per judge (default: the frame's
    judge order — training oracle left, held-out right). Top row: mean ± SE ribbons per arm, K=0
    solid + circle, K=5 dashed + square, each arm's own base as a dotted line, and a "GRPO K=5 ends"
    annotation at the censoring point. Bottom row: :func:`_delta_strip` for PTO (left-shifted) and
    GRPO (right-shifted) over the ±``oracle_noise`` band; filled = Holm p<.05, hollow = n.s. Row
    y-limits are shared across judges so the two graders read on one scale.
    (paper: ``k_contrast_headline_fig_q1q2``)
    """
    if frames is None or frames.empty or levels_long is None or levels_long.empty:
        return None
    noise = _oracle_noise(oracle_noise)
    js_list = _judges(frames, judges)
    pal = _pal(palette)
    lab = DISPLAY_NAMES.get(metric, metric)
    fig, axes = plt.subplots(2, len(js_list), figsize=(3.5 * len(js_list), 5.9), sharex=True, squeeze=False)
    for j, js in enumerate(js_list):
        ax = axes[0, j]
        lv = levels_long[(levels_long["judge"] == js) & (levels_long["metric"] == metric)]
        for arm in _ARMS:
            d = lv[lv["arm"] == arm].sort_values("iteration")
            if d.empty:
                continue
            st = K_STYLE[_k_of(arm)]
            ax.fill_between(d["iteration"], d["mean"] - d["se"], d["mean"] + d["se"],
                            color=pal[arm], alpha=0.16, lw=0)
            ax.plot(d["iteration"], d["mean"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.7,
                    color=pal[arm], label=arm_label(arm), zorder=3)
            base = d.loc[d["iteration"] == 0, "mean"]
            if len(base):
                ax.axhline(float(base.iloc[0]), ls=":", lw=0.9, color=pal[arm], alpha=0.8, zorder=1)
        ax.set_title(f"{lab} level — {js}", fontsize=10)
        ax.set_ylabel(f"{lab} (1–5), mean ± SE" if j == 0 else "")
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
        dc = lv[lv["arm"] == censor_arm]
        if len(dc):
            it_end = int(dc["iteration"].max())
            y5 = float(dc.loc[dc["iteration"] == it_end, "mean"].iloc[0])
            ax.annotate(f"{arm_label(censor_arm).replace(' (', ' ').replace(')', '')} ends",
                        xy=(it_end, y5), xytext=(it_end + 0.6, y5 + 0.32), fontsize=6.5,
                        color=pal[censor_arm], ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=pal[censor_arm], lw=0.7))
        ax = axes[1, j]
        d = frames[(frames["judge"] == js) & (frames["metric"] == metric)]
        _band(ax, noise, label=False)
        _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12)
        _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12)
        ax.set_title(f"Paired K=0 − K=5, {lab} — {js}", fontsize=10)
        ax.set_ylabel(f"Δ {lab} (K=0 − K=5), 95% CI" if j == 0 else "")
        ax.set_xlabel("iteration (0 = base vs base)")
        ax.set_xticks(range(0, int(frames["iteration"].max()) + 1))
        if j == 0:
            ax.legend(handles=_sig_handles(pal, noise), fontsize=6.5, frameon=False, loc="upper left",
                      ncol=3, handlelength=1.4, columnspacing=0.9)
    for row in (0, 1):
        ymin = min(a.get_ylim()[0] for a in axes[row]); ymax = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(ymin, ymax)
    fig.tight_layout()
    return fig


def k_delta_grid(frames: pd.DataFrame, judge: str, *, metrics: Sequence[str] = _RUBRICS,
                 palette: Optional[dict] = None, oracle_noise: Optional[float] = None, ncols: int = 3):
    """The paired K=0 − K=5 delta by iteration on EVERY rubric, one grader — a 3×3 grid.

    Point-scale rubrics get the ±``oracle_noise`` band; PCT / MICI (0-1) get a bare zero line and
    say so in their title; lower-better rubrics are flagged "↓ lower better". PTO left-shifted,
    GRPO right-shifted, filled = Holm p<.05. (paper: ``k_contrast_headline_fig_grid_{judge}``)
    """
    if frames is None or frames.empty:
        return None
    noise = _oracle_noise(oracle_noise)
    pal = _pal(palette)
    metrics = [m for m in metrics if m in set(frames["metric"])]
    if not metrics:
        return None
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.2 * nrows), sharex=True, squeeze=False)
    n_it = int(frames["iteration"].max()) + 1
    for ax, m in zip(axes.flat, metrics):
        d = frames[(frames["judge"] == judge) & (frames["metric"] == m)]
        if m in _FIVE_POINT:
            _band(ax, noise, label=False)
        else:
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
        _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.5)
        _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.5)
        title = DISPLAY_NAMES.get(m, m)
        if m in _LOWER_BETTER:
            title += " ↓ lower better"
        if m not in _FIVE_POINT:
            title += f"\n(own scale; no ±{noise:.2f} band)"
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Δ (K=0 − K=5)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, n_it))
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("iteration (0 = base vs base)", fontsize=8)
    pto_l, grpo_l = _censor_labels(frames)
    fig.legend(handles=_sig_handles(pal, noise, pto_label=pto_l, grpo_label=grpo_l), loc="upper center",
               bbox_to_anchor=(0.5, 1.03), ncol=5, fontsize=7.5, frameon=False)
    fig.suptitle(f"Paired K=0 − K=5 by iteration, all instruments — grader: {judge}", y=1.06, fontsize=10)
    fig.tight_layout()
    return fig


def k_channels_grid(channels: pd.DataFrame, channels_text: Optional[pd.DataFrame] = None, *,
                    fig_channels: Sequence[str] = ("MICI_OverPraise_rate", "MICI_AdviseNoPermission_rate",
                                                   "B6_AF_per_turn"),
                    text_channels: Sequence[str] = ("conv_len", "mean_turn_len"),
                    judges: Optional[Sequence[str]] = None, palette: Optional[dict] = None):
    """The K contrast on the behaviour channels: rows = graders, cols = the oracle-coded
    ``fig_channels`` (per therapist turn), plus a last column of judge-invariant text channels.

    ``channels`` / ``channels_text`` = :func:`eda_analysis.lookahead.channel_k_frames`'s
    ``"channels"`` / ``"channels_text"``. (paper: ``k_contrast_headline_fig_channels``)
    """
    if channels is None or channels.empty:
        return None
    pal = _pal(palette)
    js_list = _judges(channels, judges)
    fig_channels = [c for c in fig_channels if c in set(channels["metric"])]
    text_channels = ([c for c in text_channels if c in set(channels_text["metric"])]
                     if channels_text is not None and not channels_text.empty else [])
    ncols = len(fig_channels) + (1 if text_channels else 0)
    nrows = max(len(js_list), len(text_channels), 1)
    if ncols == 0:
        return None
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 2.2 * nrows), sharex=True, squeeze=False)
    n_it = int(channels["iteration"].max()) + 1
    for i, js in enumerate(js_list):
        for j, ch in enumerate(fig_channels):
            ax = axes[i, j]
            d = channels[(channels["judge"] == js) & (channels["metric"] == ch)]
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
            _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.2)
            _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.2)
            ax.set_title(f"{DISPLAY_NAMES.get(ch, ch)}{' ↓' if ch in _LOWER_BETTER else ''}\n{js}", fontsize=7.5)
            ax.set_ylabel("Δ per therapist turn (K=0 − K=5)", fontsize=7)
            ax.tick_params(labelsize=6.5)
    for i in range(len(js_list), nrows):
        for j in range(len(fig_channels)):
            axes[i, j].set_visible(False)
    if text_channels:
        for i, ch in enumerate(text_channels):
            ax = axes[i, -1]
            d = channels_text[channels_text["metric"] == ch]
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
            _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.2)
            _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.2)
            unit = {"conv_len": "utterances", "mean_turn_len": "chars / therapist turn",
                    "n_th_turns": "therapist turns", "q_per_turn": "'?' / therapist turn",
                    "loop": "fraction of convs"}.get(ch, "")
            ax.set_title(f"{DISPLAY_NAMES.get(ch, ch)}\n(text, judge-invariant)", fontsize=7.5)
            ax.set_ylabel(f"Δ {unit} (K=0 − K=5)", fontsize=7)
            ax.tick_params(labelsize=6.5)
        for i in range(len(text_channels), nrows):
            axes[i, -1].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("iteration (0 = base vs base)", fontsize=7)
        ax.set_xticks(range(0, n_it, 2))
    pto_l, grpo_l = _censor_labels(channels)
    fig.legend(handles=_sig_handles(pal, None, pto_label=pto_l, grpo_label=grpo_l), loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout()
    return fig


def _to_pairs_layout(frames: pd.DataFrame, primary: Optional[str], heldout: Optional[str]) -> pd.DataFrame:
    """Accept either :func:`eda_analysis.transfer.cross_k_pairs` (``primary_*``/``judge_*``) or the
    long :func:`eda_analysis.lookahead.paired_k_frames` (``judge`` column) and return the former."""
    if "primary_delta" in frames.columns:
        return frames
    js = _judges(frames, None)
    if len(js) < 2:
        raise ValueError("k_contrast_both_judges needs two graders in `frames`")
    primary = primary or js[0]; heldout = heldout or js[1]
    keys = ["method", "metric", "iteration"]
    ren = {"mean_delta": "delta", "dz": "dz", "ci_lo": "ci_lo", "ci_hi": "ci_hi", "p": "p", "p_holm": "p_holm", "n": "n"}
    P = frames[frames["judge"] == primary][keys + list(ren)].rename(columns={k: f"primary_{v}" for k, v in ren.items()})
    J = frames[frames["judge"] == heldout][keys + list(ren)].rename(columns={k: f"judge_{v}" for k, v in ren.items()})
    return P.merge(J, on=keys, how="inner")


def k_contrast_both_judges(frames: pd.DataFrame, *, metrics: Sequence[str] = ("Q1Q2", "MICI"),
                           methods: Sequence[str] = ("PTO", "GRPO"), palette: Optional[dict] = None,
                           primary: Optional[str] = None, heldout: Optional[str] = None,
                           primary_label: str = "training oracle (gpt-4o-mini)",
                           heldout_label: str = "held-out judge (Claude Haiku 4.5)"):
    """The K contrast under BOTH graders on one axis: rows = ``metrics``, cols = ``methods``.

    Primary: solid line + filled circle + CI ribbon; held-out: dotted line + open circle + CI bars;
    a star marks Holm p<.05 (across iterations) under either grader. Takes the transfer ``pairs``
    frame (``primary_*``/``judge_*``) or the long ``paired_k_frames`` (first judge = primary,
    second = held-out, or name them). Sign ``+ => K=0 higher``; on MICI (lower-better) + favours
    K=5. (paper: ``cross_k_multijudge_fig_kcontrast``)
    """
    if frames is None or frames.empty:
        return None
    pairs = _to_pairs_layout(frames, primary, heldout)
    pal = _pal(palette)
    metrics = [m for m in metrics if m in set(pairs["metric"])]
    if not metrics:
        return None
    n_it = int(pairs["iteration"].max()) + 1
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(3.5 * len(methods), 2.45 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, m in enumerate(metrics):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            sub = pairs[(pairs["method"] == method) & (pairs["metric"] == m)].sort_values("iteration")
            col = pal[f"{method}_LA0"]
            ax.fill_between(sub["iteration"], sub["primary_ci_lo"], sub["primary_ci_hi"], color=col, alpha=0.13, lw=0)
            ax.plot(sub["iteration"], sub["primary_delta"], ls="-", marker="o", color=col, lw=1.6, ms=5.5)
            ax.errorbar(sub["iteration"], sub["judge_delta"],
                        yerr=[sub["judge_delta"] - sub["judge_ci_lo"], sub["judge_ci_hi"] - sub["judge_delta"]],
                        fmt="none", ecolor=col, elinewidth=0.9, capsize=2, alpha=0.8)
            ax.plot(sub["iteration"], sub["judge_delta"], ls=":", marker="o", mfc="white", mec=col, color=col, lw=1.6, ms=5.5)
            for pref in ("primary", "judge"):
                sig = sub[sub[f"{pref}_p_holm"] < 0.05]
                if len(sig):
                    ax.scatter(sig["iteration"], sig[f"{pref}_delta"], marker="*", s=75, color=col, zorder=5,
                               edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="0.35", lw=0.8)
            censored = (not sub.empty) and int(sub["iteration"].max()) < n_it - 1
            ax.set_title(f"{method}: {m}, K=0 − K=5, both graders"
                         + (f" (LA5 ends at {int(sub['iteration'].max())})" if censored else ""), fontsize=9.5)
            unit = "rate" if m in ("PCT", "MICI") else "score points"
            ax.set_ylabel(f"{m} Δ (K=0 − K=5), {unit}" if j == 0 else "", fontsize=9)
            ax.set_xticks(range(0, n_it))
            ax.tick_params(labelsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel("iteration (0 = base: two independent draws)", fontsize=9)
            ax.grid(True, alpha=0.35)
    h = [Line2D([], [], color="0.2", ls="-", marker="o", ms=5.5, label=primary_label + ", CI ribbon"),
         Line2D([], [], color="0.2", ls=":", marker="o", mfc="white", ms=5.5, label=heldout_label + ", CI bars"),
         Line2D([], [], color="0.2", ls="none", marker="*", ms=9, mec="black", mew=0.5, label="Holm p < .05 (across iterations)")]
    lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8,
                    title="sign: + => K=0 higher.  MICI is lower-is-better, so on MICI + favours K=5.  Paired on persona (n = 96).",
                    title_fontsize=7.5)
    lg.get_title().set_color("0.3")
    fig.tight_layout()
    return fig


def k_retention(retention: pd.DataFrame, *, metrics: Sequence[str] = ("Q1", "Q2", "MICI"),
                methods: Sequence[str] = ("PTO", "GRPO"), ref_kind: str = "own_base",
                palette: Optional[dict] = None, ylim_default=(-0.4, 2.0), ylim_mici=(-0.5, 6.0),
                heldout_label: str = "Claude Haiku 4.5", primary_label: str = "gpt-4o-mini (training oracle)"):
    """Gain retention (Δ held-out / Δ primary over the arm's OWN base) by K: rows = ``metrics``,
    cols = ``methods``; K=0 solid + circle, K=5 dashed + square, persona-bootstrap CI ribbons,
    ``retention = 1`` dash-dotted (the gain is fully real to the held-out judge). Blank where the
    primary delta sits under the floor. ``retention`` = :func:`eda_analysis.transfer.retention_by_k`'s
    ``"retention"``. (paper: ``cross_k_multijudge_fig_retention``)
    """
    if retention is None or retention.empty:
        return None
    pal = _pal(palette)
    ret = retention[retention["ref_kind"] == ref_kind]
    metrics = [m for m in metrics if m in set(ret["metric"])]
    if not metrics:
        return None
    n_it = int(ret["iteration"].max()) + 1
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(3.5 * len(methods), 2.2 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, m in enumerate(metrics):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            censored = False
            for K in (0, 5):
                arm = f"{method}_LA{K}"
                sub = ret[(ret["arm"] == arm) & (ret["metric"] == m)].sort_values("iteration")
                if sub.empty:
                    continue
                st = K_STYLE[K]
                ax.plot(sub["iteration"], sub["retention"], ls=st["ls"], marker=st["marker"], color=pal[arm], lw=1.7,
                        ms=5.5, label=arm.replace("_LA", " K="))
                ax.fill_between(sub["iteration"], sub["retention_ci_lo"], sub["retention_ci_hi"], color=pal[arm],
                                alpha=0.15, lw=0)
                if K == 5 and int(sub["iteration"].max()) < n_it - 1:
                    censored = int(sub["iteration"].max())
            ax.axhline(1.0, color="0.25", lw=0.9, ls="-.")
            ax.axhline(0.0, color="0.6", lw=0.7)
            ax.set_title(f"{method}: {m} gain retention" + (" (harm channel)" if m in _LOWER_BETTER else "")
                         + (f" (LA5 ends at {censored})" if censored is not False else ""), fontsize=9.5)
            ax.set_ylabel("retention = Δ held-out / Δ primary" if j == 0 else "", fontsize=9)
            ax.set_xticks(range(0, n_it))
            ax.set_ylim(ylim_mici if m == "MICI" else ylim_default)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.35)
            if i == len(metrics) - 1:
                ax.set_xlabel("iteration (gain over the arm's OWN base)", fontsize=9)
    h = [Line2D([], [], color=pal[a], ls=K_STYLE[_k_of(a)]["ls"], marker=K_STYLE[_k_of(a)]["marker"], ms=5.5, lw=1.7,
                label=a.replace("_LA", " K=")) for a in _ARMS]
    h.append(Line2D([], [], color="0.25", ls="-.", lw=0.9, label="retention = 1 (gain fully real to the held-out judge)"))
    floors = sorted(set(ret["min_primary_delta"].dropna())) if "min_primary_delta" in ret.columns else []
    floor_txt = (f"blank where |Δ primary| < {floors[-1]:.2f} (point-scale rubrics) or < {floors[0]:.2f} (PCT / MICI)"
                 if len(floors) == 2 else "blank where |Δ primary| is under the floor")
    lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=8,
                    title=f"Δ held-out = {heldout_label} gain over base;  Δ primary = {primary_label} gain over base;  "
                          f"ribbons = persona-bootstrap 95% CI;  {floor_txt}", title_fontsize=7.2)
    lg.get_title().set_color("0.3")
    fig.tight_layout()
    return fig


def k_did(did: pd.DataFrame, method_gap: pd.DataFrame, *, metric: str = "Q1Q2",
          judges: Optional[Sequence[str]] = None, judge_titles: Optional[dict] = None,
          gap_colors: dict = None, did_color: str = "#CC79A7"):
    """The method gap PTO − GRPO at each K (top) over the K × method DiD (bottom), one column per grader.

    ``method_gap`` / ``did`` = :func:`eda_analysis.lookahead.method_gap_by_iter` /
    :func:`~eda_analysis.lookahead.did_by_iter`. Gap: K=0 black solid + circle, K=5 green dashed +
    square, CI ribbons, stars = Holm p<.05 across iterations. DiD (= gap(K=0) − gap(K=5), + => PTO's
    lead is larger at K=0): one series with its dz annotated per iteration; estimable only while all
    four arms run (to 5). Row y-limits shared across graders. (paper: ``cross_k_multijudge_fig_did``)
    """
    if did is None or did.empty or method_gap is None or method_gap.empty:
        return None
    gap_colors = gap_colors or {0: "#111111", 5: "#009E73"}
    js_list = _judges(method_gap, judges)
    judge_titles = judge_titles or {}
    from ..constants import judge_dirname, PRIMARY_JUDGE_TAG
    prim = judge_dirname(PRIMARY_JUDGE_TAG)
    n_it = int(method_gap["iteration"].max()) + 1
    fig, axes = plt.subplots(2, len(js_list), figsize=(3.5 * len(js_list), 5.2), sharex=True, squeeze=False)
    for j, jn in enumerate(js_list):
        title = judge_titles.get(jn) or (f"{jn} (training oracle)" if jn == prim else f"{jn} (held-out)")
        ax = axes[0, j]
        for K in (0, 5):
            sub = method_gap[(method_gap["judge"] == jn) & (method_gap["K"] == K) & (method_gap["metric"] == metric)].sort_values("iteration")
            if sub.empty:
                continue
            st = K_STYLE[K]
            ax.fill_between(sub["iteration"], sub["ci_lo"], sub["ci_hi"], color=gap_colors[K], alpha=0.13, lw=0)
            ax.plot(sub["iteration"], sub["delta"], ls=st["ls"], marker=st["marker"], color=gap_colors[K], lw=1.7, ms=5.5)
            sig = sub[sub["p_holm"] < 0.05]
            ax.scatter(sig["iteration"], sig["delta"], marker="*", s=75, color=gap_colors[K], zorder=5, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_title(f"{metric} gap PTO − GRPO: {title}", fontsize=9.5)
        ax.set_ylabel(f"{metric} Δ (PTO − GRPO), score points" if j == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.35)
        ax = axes[1, j]
        sub = did[(did["judge"] == jn) & (did["metric"] == metric)].sort_values("iteration")
        ax.fill_between(sub["iteration"], sub["did_ci_lo"], sub["did_ci_hi"], color=did_color, alpha=0.15, lw=0)
        ax.plot(sub["iteration"], sub["did_mean"], ls="-", marker="D", color=did_color, lw=1.7, ms=5.5)
        sig = sub[sub["p_holm"] < 0.05]
        ax.scatter(sig["iteration"], sig["did_mean"], marker="*", s=85, color=did_color, zorder=5, edgecolor="black", linewidth=0.5)
        for _, r in sub.iterrows():
            ax.annotate(f"dz {r['did_dz']:.2f}", (r["iteration"], r["did_ci_hi"]), textcoords="offset points",
                        xytext=(0, 3), ha="left" if r["iteration"] == 0 else "center", fontsize=6.5, color="0.25")
        ax.axhline(0, color="0.35", lw=0.8)
        did_end = int(sub["iteration"].max()) if len(sub) else n_it - 1
        ax.set_title(f"{metric} DiD: {title}", fontsize=9.5)
        ax.set_ylabel(f"{metric} DiD, score points" if j == 0 else "", fontsize=9)
        ax.set_xlabel(f"iteration (0 = base draws; DiD estimable only to {did_end})", fontsize=9)
        ax.set_xticks(range(0, n_it))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.35)
    for row in (0, 1):
        ylo = min(a.get_ylim()[0] for a in axes[row]); yhi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(ylo, yhi + (0.12 if row == 1 else 0))
    k5_end = int(method_gap.loc[method_gap["K"] == 5, "iteration"].max()) if (method_gap["K"] == 5).any() else None
    h = [Line2D([], [], color=gap_colors[0], ls="-", marker="o", ms=5.5, lw=1.7, label="K=0: PTO_LA0 − GRPO_LA0"),
         Line2D([], [], color=gap_colors[5], ls="--", marker="s", ms=5.5, lw=1.7,
                label="K=5: PTO_LA5 − GRPO_LA5" + (f" (to iter {k5_end})" if k5_end is not None else "")),
         Line2D([], [], color=did_color, ls="-", marker="D", ms=5.5, lw=1.7, label="DiD = gap(K=0) − gap(K=5)"),
         Line2D([], [], color="0.2", ls="none", marker="*", ms=9, mec="black", mew=0.5, label="Holm p < .05 (across iterations)")]
    lg = fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=8,
                    title="signs: gap + => PTO higher;  DiD + => PTO's lead over GRPO is larger at K=0 than at K=5.  "
                          "Ribbons = persona-bootstrap 95% CI, n = 96 personas.", title_fontsize=7.5)
    lg.get_title().set_color("0.3")
    fig.tight_layout()
    return fig
