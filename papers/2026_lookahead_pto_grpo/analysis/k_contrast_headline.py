"""k_contrast_headline.py — THE headline artifact of the look-ahead paper.

The four-arm, persona-paired K contrast (K=0 vs K=5, within method) under BOTH graders in ONE
frame — the training oracle (gpt-4o-mini) side by side with the held-out judge (Claude Haiku 4.5) —
for every matched iteration 0..10 (GRPO: 0..5, right-censored), every rubric, plus the oracle-coded
behaviour channels (MICI / MITI, per turn and per session) and the deterministic text channels.

Conventions (restated in every caption):
  * sign: ``+ delta => K=0 higher`` (mirrors ``eda_analysis.stats.paired_k_comparison``);
  * pairing unit: ``persona_id`` (the 96 patient personas recur in every model state);
  * Holm: within a (judge, method, metric) family ACROSS iterations 0..N — a different family
    than the tracked ``k_paired_by_method.md`` (which corrects across rubrics within one
    iteration), so ``p`` agrees with that table cell-for-cell while ``p_holm`` need not;
  * iteration 0 = two INDEPENDENT base draws (K=0-arm base vs K=5-arm base) — a free noise-floor
    row, computed with the same machinery;
  * the two graders' raw scores are never averaged.

Outputs (all prefixed ``k_contrast_headline_``):
  tables/   <method>_<judge>.md (long: metric x iteration), table1.md (Q1Q2 compact) +
            table1_{Q1,Q2,MICI,PCT}.md, summary.md, levels.md (+ levels_long), channels_<method>_<judge>.md,
            channels_text_<method>.md, channels_summary.md
  figures/  fig_q1q2.png, fig_grid_primary.png, fig_grid_heldout.png, fig_channels.png
  analysis/out/k_contrast_headline.json  (the ledger)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import eda_analysis  # noqa: E402
from eda_analysis import behavior  # noqa: E402
from eda_analysis.constants import set_active_judge, DISPLAY_NAMES, LOWER_IS_BETTER as _LIB  # noqa: E402

SCRIPT = "k_contrast_headline"
METHODS = ["PTO", "GRPO"]
JUDGE_KEYS = ["primary", "heldout"]
JS = {"primary": C.JUDGE_SHORT[C.PRIMARY], "heldout": C.JUDGE_SHORT[C.HELDOUT]}   # short labels
JL = {"primary": C.JUDGE_LABEL[C.PRIMARY], "heldout": C.JUDGE_LABEL[C.HELDOUT]}   # long labels
ORACLE_NOISE = C.EdaConfig().oracle_noise                                            # ±0.10 band
LOWER_BETTER = set(C.LOWER_IS_BETTER) | set(_LIB)
TEXT_CHANNELS = ["conv_len", "n_th_turns", "mean_turn_len", "q_per_turn", "loop"]
FIG_CHANNELS = ["MICI_OverPraise_rate", "MICI_AdviseNoPermission_rate", "B6_AF_per_turn"]
FIVE_POINT = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI"]   # rubrics on the 1-5 scale the ±0.10 band refers to

SIGN_NOTE = "Sign: + = K=0 higher (K=0 minus K=5). Paired on persona_id (96 personas)."
CENSOR_NOTE = "GRPO_LA5 is right-censored at iteration 5, so GRPO rows stop at 5."
HOLM_NOTE = ("p_holm = Holm across iterations 0..N within each (judge, method, metric); "
             "iteration 0 = two independent base draws (noise floor).")


def model_name(method: str, K: int, it: int) -> str:
    return f"{method}Exp3_LA{K}_{'Base' if it == 0 else f'I{it}'}"


def stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def label_of(metric: str) -> str:
    lab = DISPLAY_NAMES.get(metric, metric)
    if metric == "loop":
        lab = "Degenerate loop (fraction of convs)"
    if metric in LOWER_BETTER:
        lab += " [lower better]"
    return lab


# ── core: paired K contrast over a long frame ─────────────────────────────────

def k_contrast_long(long: pd.DataFrame, metrics, *, judge_key: str) -> pd.DataFrame:
    """Per (method, metric, iteration): paired K0-K5 contrast + both levels. Holm within
    (method, metric) across iterations."""
    rows = []
    for method in METHODS:
        for m in metrics:
            W = C.wide(long, m)
            if W.empty:
                continue
            iters = sorted({int(it) for it in long.loc[long["arm"] == f"{method}_LA5", "iteration"]}
                           & {int(it) for it in long.loc[long["arm"] == f"{method}_LA0", "iteration"]})
            for it in iters:
                a, b = model_name(method, 0, it), model_name(method, 5, it)
                if a not in W.columns or b not in W.columns:
                    continue
                r = C.paired(W[a].to_numpy(), W[b].to_numpy())
                rows.append({"judge": JS[judge_key], "method": method, "metric": m, "iteration": it,
                             "mean_K0": float(W[a].mean()), "mean_K5": float(W[b].mean()),
                             "se_K0": float(W[a].std(ddof=1) / np.sqrt(W[a].notna().sum())),
                             "se_K5": float(W[b].std(ddof=1) / np.sqrt(W[b].notna().sum())),
                             **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    for (_, _), idx in df.groupby(["method", "metric"]).groups.items():
        df.loc[idx, "p_holm"] = C.holm(df.loc[idx, "p"].to_numpy())
    df["sig"] = df["p_holm"].map(stars)
    df["lower_better"] = df["metric"].isin(LOWER_BETTER)
    return df[["judge", "method", "metric", "iteration", "n", "mean_K0", "mean_K5", "se_K0", "se_K5",
               "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "sig", "lower_better"]]


def levels_long(long: pd.DataFrame, metrics, *, judge_key: str) -> pd.DataFrame:
    d = long[long["questionnaire"].isin(metrics)]
    g = (d.groupby(["arm", "questionnaire", "iteration"])["score"]
         .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count").reset_index())
    g["se"] = g["sd"] / np.sqrt(g["n"])
    g["judge"] = JS[judge_key]
    g = g.rename(columns={"questionnaire": "metric"})
    return g[["judge", "arm", "metric", "iteration", "n", "mean", "sd", "se"]]


def summarize(df: pd.DataFrame, *, group_cols=("judge", "method", "metric")) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(group_cols)):
        g = g.sort_values("iteration")
        trained = g[g["iteration"] >= 1]
        sig = g[g["p_holm"] < 0.05]
        lb = bool(g["lower_better"].iloc[0])
        pos, neg = sig[sig["mean_delta"] > 0], sig[sig["mean_delta"] < 0]
        k0_better, k5_better = (neg, pos) if lb else (pos, neg)
        imax = g.loc[g["dz"].abs().idxmax()] if g["dz"].notna().any() else None
        base = g[g["iteration"] == 0]
        rows.append({**dict(zip(group_cols, key)),
                     "n_iters": int(len(g)),
                     "n_sig_K0_higher": int(len(pos)), "n_sig_K5_higher": int(len(neg)),
                     "n_sig_K0_better": int(len(k0_better)), "n_sig_K5_better": int(len(k5_better)),
                     "iters_sig_K0_higher": ",".join(str(int(i)) for i in pos["iteration"]),
                     "iters_sig_K5_higher": ",".join(str(int(i)) for i in neg["iteration"]),
                     "mean_delta_iters1toN": float(trained["mean_delta"].mean()) if len(trained) else np.nan,
                     "mean_dz_iters1toN": float(trained["dz"].mean()) if len(trained) else np.nan,
                     "base_delta": float(base["mean_delta"].iloc[0]) if len(base) else np.nan,
                     "base_dz": float(base["dz"].iloc[0]) if len(base) else np.nan,
                     "max_abs_dz": float(abs(imax["dz"])) if imax is not None else np.nan,
                     "max_abs_dz_iter": int(imax["iteration"]) if imax is not None else -1,
                     "max_abs_dz_delta": float(imax["mean_delta"]) if imax is not None else np.nan,
                     "lower_better": lb})
    return pd.DataFrame(rows)


def compact_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """rows = iteration, columns = method x judge: 'delta (dz)stars'."""
    d = df[df["metric"] == metric]
    out = pd.DataFrame({"iteration": list(range(0, 11))})
    for method in METHODS:
        for jk in JUDGE_KEYS:
            col = f"{method} · {JS[jk]}"
            cell = {}
            for _, r in d[(d["method"] == method) & (d["judge"] == JS[jk])].iterrows():
                cell[int(r["iteration"])] = f"{r['mean_delta']:+.3f} ({r['dz']:+.2f}){r['sig']}"
            out[col] = [cell.get(i, "—") for i in out["iteration"]]
    return out


# ── figures ───────────────────────────────────────────────────────────────────

def _delta_strip(ax, df, method, color, xoff, *, ms=5.5, lw=1.4, capsize=2.2):
    d = df[df["method"] == method].sort_values("iteration")
    if d.empty:
        return
    x = d["iteration"].to_numpy(float) + xoff
    y = d["mean_delta"].to_numpy(float)
    yerr = np.vstack([y - d["ci_lo"].to_numpy(float), d["ci_hi"].to_numpy(float) - y])
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=lw, capsize=capsize, zorder=2)
    ax.plot(x, y, ls="-", lw=1.0, color=color, alpha=0.55, zorder=2)
    sig = (d["p_holm"] < 0.05).to_numpy()
    ax.plot(x[sig], y[sig], "o", ms=ms, mfc=color, mec=color, zorder=3)
    ax.plot(x[~sig], y[~sig], "o", ms=ms, mfc="white", mec=color, mew=1.4, zorder=3)


def _band(ax, label=True):
    ax.axhline(0, color="#333333", lw=0.9, zorder=1)
    ax.axhspan(-ORACLE_NOISE, ORACLE_NOISE, color="#999999", alpha=0.18, lw=0, zorder=0)
    if label:
        ax.text(0.99, ORACLE_NOISE, f" ±{ORACLE_NOISE:.2f} oracle repeatability",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.5,
                color="#555555")


def fig_q1q2(K: pd.DataFrame, LV: pd.DataFrame, pal: dict):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.9), sharex=True)
    for j, jk in enumerate(JUDGE_KEYS):
        js = JS[jk]
        # top: levels
        ax = axes[0, j]
        lv = LV[(LV["judge"] == js) & (LV["metric"] == "Q1Q2")]
        for arm in C.ARMS:
            d = lv[lv["arm"] == arm].sort_values("iteration")
            if d.empty:
                continue
            k = C.k_of(arm); st = C.K_STYLE[k]
            ax.fill_between(d["iteration"], d["mean"] - d["se"], d["mean"] + d["se"],
                            color=pal[arm], alpha=0.16, lw=0)
            ax.plot(d["iteration"], d["mean"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.7,
                    color=pal[arm], label=eda_analysis.arm_label(arm), zorder=3)
            base = d.loc[d["iteration"] == 0, "mean"]
            if len(base):
                ax.axhline(float(base.iloc[0]), ls=":", lw=0.9, color=pal[arm], alpha=0.8, zorder=1)
        ax.set_title(f"Q1+Q2 level — {js}", fontsize=10)
        ax.set_ylabel("Q1+Q2 (1–5), mean ± SE" if j == 0 else "")
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
        g5 = lv[(lv["arm"] == "GRPO_LA5") & (lv["iteration"] == 5)]
        if len(g5):
            y5 = float(g5["mean"].iloc[0])
            ax.annotate("GRPO K=5 ends", xy=(5, y5), xytext=(5.6, y5 + 0.32), fontsize=6.5,
                        color=pal["GRPO_LA5"], ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=pal["GRPO_LA5"], lw=0.7))
        # bottom: paired delta
        ax = axes[1, j]
        d = K[(K["judge"] == js) & (K["metric"] == "Q1Q2")]
        _band(ax, label=False)
        _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12)
        _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12)
        ax.set_title(f"Paired K=0 − K=5, Q1+Q2 — {js}", fontsize=10)
        ax.set_ylabel("Δ Q1+Q2 (K=0 − K=5), 95% CI" if j == 0 else "")
        ax.set_xlabel("iteration (0 = base vs base)")
        ax.set_xticks(range(0, 11))
        if j == 0:
            handles = [Line2D([], [], color=pal["PTO_LA0"], marker="o", ls="-", label="PTO"),
                       Line2D([], [], color=pal["GRPO_LA0"], marker="o", ls="-", label="GRPO"),
                       Line2D([], [], color="#444444", marker="o", ls="", mfc="#444444", label="Holm p<.05"),
                       Line2D([], [], color="#444444", marker="o", ls="", mfc="white", label="n.s."),
                       plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.25,
                                     label=f"±{ORACLE_NOISE:.2f} oracle repeatability")]
            ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", ncol=3,
                      handlelength=1.4, columnspacing=0.9)
    ymin = min(a.get_ylim()[0] for a in axes[1]); ymax = max(a.get_ylim()[1] for a in axes[1])
    for a in axes[1]:
        a.set_ylim(ymin, ymax)
    ymin = min(a.get_ylim()[0] for a in axes[0]); ymax = max(a.get_ylim()[1] for a in axes[0])
    for a in axes[0]:
        a.set_ylim(ymin, ymax)
    return fig


def fig_grid(K: pd.DataFrame, jk: str, pal: dict):
    js = JS[jk]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True)
    for ax, m in zip(axes.flat, C.RUBRICS):
        d = K[(K["judge"] == js) & (K["metric"] == m)]
        if m in FIVE_POINT:
            _band(ax, label=False)
        else:
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
        _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.5)
        _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.5)
        title = DISPLAY_NAMES.get(m, m)
        if m in LOWER_BETTER:
            title += " ↓ lower better"
        if m not in FIVE_POINT:
            title += "\n(own scale; no ±0.10 band)"
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Δ (K=0 − K=5)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, 11))
    for ax in axes[-1]:
        ax.set_xlabel("iteration (0 = base vs base)", fontsize=8)
    handles = [Line2D([], [], color=pal["PTO_LA0"], marker="o", ls="-", label="PTO (iters 0–10)"),
               Line2D([], [], color=pal["GRPO_LA0"], marker="o", ls="-", label="GRPO (iters 0–5, censored)"),
               Line2D([], [], color="#444444", marker="o", ls="", mfc="#444444", label="Holm p<.05"),
               Line2D([], [], color="#444444", marker="o", ls="", mfc="white", label="n.s."),
               plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.25, label=f"±{ORACLE_NOISE:.2f} oracle repeatability")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5, fontsize=7.5,
               frameon=False)
    fig.suptitle(f"Paired K=0 − K=5 by iteration, all instruments — grader: {js}", y=1.06, fontsize=10)
    return fig


def fig_channels(KC: pd.DataFrame, KT: pd.DataFrame, pal: dict):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True)
    for i, jk in enumerate(JUDGE_KEYS):
        js = JS[jk]
        for j, ch in enumerate(FIG_CHANNELS):
            ax = axes[i, j]
            d = KC[(KC["judge"] == js) & (KC["metric"] == ch)]
            ax.axhline(0, color="#333333", lw=0.9, zorder=1)
            _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.2)
            _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.2)
            ax.set_title(f"{DISPLAY_NAMES.get(ch, ch)}{' ↓' if ch in LOWER_BETTER else ''}\n{js}",
                         fontsize=7.5)
            ax.set_ylabel("Δ per therapist turn (K=0 − K=5)", fontsize=7)
            ax.tick_params(labelsize=6.5)
    # 4th column: judge-invariant text channels
    for i, ch in enumerate(["conv_len", "mean_turn_len"]):
        ax = axes[i, 3]
        d = KT[KT["metric"] == ch]
        ax.axhline(0, color="#333333", lw=0.9, zorder=1)
        _delta_strip(ax, d, "PTO", pal["PTO_LA0"], -0.12, ms=4.2)
        _delta_strip(ax, d, "GRPO", pal["GRPO_LA0"], +0.12, ms=4.2)
        unit = "utterances" if ch == "conv_len" else "chars / therapist turn"
        ax.set_title(f"{DISPLAY_NAMES.get(ch, ch)}\n(text, judge-invariant)", fontsize=7.5)
        ax.set_ylabel(f"Δ {unit} (K=0 − K=5)", fontsize=7)
        ax.tick_params(labelsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("iteration (0 = base vs base)", fontsize=7)
        ax.set_xticks(range(0, 11, 2))
    handles = [Line2D([], [], color=pal["PTO_LA0"], marker="o", ls="-", label="PTO (iters 0–10)"),
               Line2D([], [], color=pal["GRPO_LA0"], marker="o", ls="-", label="GRPO (iters 0–5, censored)"),
               Line2D([], [], color="#444444", marker="o", ls="", mfc="#444444", label="Holm p<.05"),
               Line2D([], [], color="#444444", marker="o", ls="", mfc="white", label="n.s.")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=7.5,
               frameon=False)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    C.style()
    pal = C.palette()
    L = C.Ledger(SCRIPT)

    # 1) rubric scores under both graders (ONE call; primary then held-out; ends on primary)
    S = C.load_scores_both()

    # 2) behaviour channels: oracle-coded channels are judge-dependent (read from that judge's
    #    MITI/MICI partition); text channels are judge-invariant (computed once).
    cfg = C.EdaConfig(view="L5", verbose=False)
    arms = eda_analysis.cross_k_arms(cfg)
    CH = {}
    for jk in JUDGE_KEYS:
        set_active_judge("" if jk == "primary" else C.HELDOUT, 0)
        ch = behavior.channel_scores_long(arms)
        CH[jk] = ch[~ch["questionnaire"].isin(behavior.TEXT_CHANNELS)].copy()
    set_active_judge("", 0)
    tm = behavior.text_metrics(arms, attach_persona=True)
    tm["loop"] = tm["loop"].astype(float)
    TXT = tm.melt(id_vars=["arm", "method", "K", "model", "iteration", "is_base", "file_index", "persona_id"],
                  value_vars=TEXT_CHANNELS, var_name="questionnaire", value_name="score").dropna(subset=["score"])
    oracle_channels = [c for c in behavior.BEHAVIOR_CHANNELS if c not in behavior.TEXT_CHANNELS]

    # 3) contrasts + levels
    K = pd.concat([k_contrast_long(S[jk], C.RUBRICS, judge_key=jk) for jk in JUDGE_KEYS], ignore_index=True)
    LV = pd.concat([levels_long(S[jk], C.RUBRICS, judge_key=jk) for jk in JUDGE_KEYS], ignore_index=True)
    KC = pd.concat([k_contrast_long(CH[jk], oracle_channels, judge_key=jk) for jk in JUDGE_KEYS],
                   ignore_index=True)
    KT = k_contrast_long(TXT, TEXT_CHANNELS, judge_key="primary").assign(judge="text (judge-invariant)")

    # ── cross-check against eda_analysis.stats.paired_k_comparison (primary, PTO) ──
    ref = eda_analysis.stats.paired_k_comparison(S["primary"], "PTO", metrics=C.RUBRICS)
    ref = ref[(ref["metric"] == "Q1Q2") & (ref["iteration"] == 6)].iloc[0]
    mine = K[(K["judge"] == JS["primary"]) & (K["method"] == "PTO") & (K["metric"] == "Q1Q2")
             & (K["iteration"] == 6)].iloc[0]
    print(f"[xcheck] PTO Q1Q2 iter6 primary: eda mean_delta={ref['mean_delta']:.3f} dz={ref['dz']:.3f} "
          f"p={ref['p']:.4f} | mine mean_delta={mine['mean_delta']:.3f} dz={mine['dz']:.3f} p={mine['p']:.4f}")
    assert abs(ref["mean_delta"] - mine["mean_delta"]) < 1e-9 and abs(ref["dz"] - mine["dz"]) < 1e-9
    assert abs(round(mine["mean_delta"], 3) - 0.257) < 1e-9 and abs(round(mine["dz"], 3) - 0.417) < 1e-9
    L.put("xcheck.pto_q1q2_iter6_primary", {"mean_delta": mine["mean_delta"], "dz": mine["dz"], "p": mine["p"],
                                            "eda_mean_delta": ref["mean_delta"], "eda_dz": ref["dz"]},
          source="Exp3 eda/results/L5/tables/7_stats/gpt-4o-mini/k_paired_by_method.md (PTO, iter 6, Q1Q2)")
    refc = eda_analysis.stats.paired_k_comparison(CH["primary"], "PTO", metrics=behavior.MICI_RATE_CHANNELS)
    refc = refc[(refc["metric"] == "MICI_Severity") & (refc["iteration"] == 0)].iloc[0]
    minec = KC[(KC["judge"] == JS["primary"]) & (KC["method"] == "PTO") & (KC["metric"] == "MICI_Severity")
               & (KC["iteration"] == 0)].iloc[0]
    print(f"[xcheck] PTO MICI_Severity iter0 primary: eda {refc['mean_delta']:.3f}/{refc['dz']:.3f} | "
          f"mine {minec['mean_delta']:.3f}/{minec['dz']:.3f}")
    assert abs(refc["mean_delta"] - minec["mean_delta"]) < 1e-9

    # ── tables ──
    for method in METHODS:
        for jk in JUDGE_KEYS:
            d = K[(K["method"] == method) & (K["judge"] == JS[jk])].drop(columns=["judge", "method"])
            C.save_table(d, f"{SCRIPT}_{method.lower()}_{jk}",
                         caption=(f"**{method}, K=0 vs K=5, grader = {JL[jk]}.** Persona-paired contrast per "
                                  f"rubric × iteration: levels (mean ± SE over personas per arm), mean_delta, "
                                  f"Cohen's dz, bootstrap 95% CI, Wilcoxon p, p_holm. {SIGN_NOTE} {HOLM_NOTE} "
                                  f"MICI is lower-better (+ = K=0 WORSE there). "
                                  + (CENSOR_NOTE if method == "GRPO" else "")))
    C.save_table(K, f"{SCRIPT}_all_long",
                 caption=f"**All four method × grader K contrasts, long form.** {SIGN_NOTE} {HOLM_NOTE} {CENSOR_NOTE}")

    t1 = compact_table(K, "Q1Q2")
    C.save_table(t1, f"{SCRIPT}_table1",
                 caption=(f"**Table 1 — paired K=0 − K=5 on the training reward Q1+Q2, by iteration, under both "
                          f"graders.** Cell = mean_delta (Cohen's dz) with Holm stars (* <.05, ** <.01, *** <.001; "
                          f"{HOLM_NOTE}). {SIGN_NOTE} {CENSOR_NOTE} '—' = no matched K=5 model state."))
    for m in ["Q1", "Q2", "MICI", "PCT"]:
        C.save_table(compact_table(K, m), f"{SCRIPT}_table1_{m}",
                     caption=(f"**Paired K=0 − K=5 on {DISPLAY_NAMES.get(m, m)}, by iteration, both graders.** "
                              f"Cell = mean_delta (dz) + Holm stars. {SIGN_NOTE} {HOLM_NOTE} {CENSOR_NOTE}"
                              + (" MICI is LOWER-better: + means K=0 is MORE MI-inconsistent (worse)." if m == "MICI" else "")))

    SUM = summarize(K)
    C.save_table(SUM, f"{SCRIPT}_summary",
                 caption=(f"**Per (grader, method, rubric) summary of the K contrast across iterations.** "
                          f"n_sig_K0_higher / n_sig_K5_higher count iterations with Holm p<.05 and delta >0 / <0; "
                          f"the *_better columns flip the sign for lower-better rubrics (MICI). mean_delta_iters1toN "
                          f"averages the per-iteration paired deltas over trained iterations only; base_delta is the "
                          f"iteration-0 base-vs-base draw. {SIGN_NOTE} {HOLM_NOTE} {CENSOR_NOTE}"))

    lvq = LV[LV["metric"] == "Q1Q2"]
    lv_wide = pd.DataFrame({"iteration": list(range(0, 11))})
    for jk in JUDGE_KEYS:
        for arm in C.ARMS:
            d = lvq[(lvq["judge"] == JS[jk]) & (lvq["arm"] == arm)].set_index("iteration")
            lv_wide[f"{arm} · {JS[jk]}"] = [d["mean"].get(i, np.nan) for i in lv_wide["iteration"]]
    C.save_table(lv_wide, f"{SCRIPT}_levels",
                 caption=("**Q1+Q2 arm means by iteration under both graders** (96 conversations per cell; "
                          "iteration 0 = each arm's own base draw). Not paired — read dz/p off the contrast "
                          f"tables. {CENSOR_NOTE} GRPO_LA0 continues to iteration 10 unmatched."))
    C.save_table(LV, f"{SCRIPT}_levels_long",
                 caption="**Arm × rubric × iteration levels (mean, sd, SE over the 96 personas) under both graders.**")

    # channels
    for method in METHODS:
        for jk in JUDGE_KEYS:
            d = KC[(KC["method"] == method) & (KC["judge"] == JS[jk])].drop(columns=["judge", "method"])
            C.save_table(d, f"{SCRIPT}_channels_{method.lower()}_{jk}",
                         caption=(f"**{method}, K=0 vs K=5 on the oracle-coded behaviour channels, grader = {JL[jk]}.** "
                                  f"MICI channels (per therapist turn `_rate`, per session counts) are lower-better; "
                                  f"MITI channels (`_per_turn`, counts, RtoQ, %CR, %MICO) are MI-consistent behaviours. "
                                  f"{SIGN_NOTE} {HOLM_NOTE} " + (CENSOR_NOTE if method == "GRPO" else "")))
        d = KT[KT["method"] == method].drop(columns=["judge", "method"])
        C.save_table(d, f"{SCRIPT}_channels_text_{method.lower()}",
                     caption=(f"**{method}, K=0 vs K=5 on the deterministic text channels (judge-invariant).** "
                              f"conv_len = utterances per conversation; n_th_turns = therapist turns; mean_turn_len = "
                              f"characters per therapist turn; q_per_turn = literal '?' per therapist turn; loop = "
                              f"fraction of conversations with a verbatim repeated therapist turn. {SIGN_NOTE} {HOLM_NOTE} "
                              + (CENSOR_NOTE if method == "GRPO" else "")))
    KCS = summarize(pd.concat([KC, KT], ignore_index=True))
    C.save_table(KCS, f"{SCRIPT}_channels_summary",
                 caption=(f"**Per (grader, method, channel) summary of the behaviour-channel K contrast.** "
                          f"Same columns as the rubric summary; MICI channels and their counts are lower-better. "
                          f"{SIGN_NOTE} {HOLM_NOTE} {CENSOR_NOTE}"))

    # ── figures ──
    p = C.save_fig(fig_q1q2(K, LV, pal), f"{SCRIPT}_fig_q1q2"); print("fig", p)
    for jk in JUDGE_KEYS:
        p = C.save_fig(fig_grid(K, jk, pal), f"{SCRIPT}_fig_grid_{jk}"); print("fig", p)
    p = C.save_fig(fig_channels(KC, KT, pal), f"{SCRIPT}_fig_channels"); print("fig", p)

    # ── ledger ──
    def _row(r):
        return {k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else r[k])
                for k in ["n", "mean_K0", "mean_K5", "mean_delta", "dz", "ci_lo", "ci_hi", "p", "p_holm", "sig"]}
    for m in ["Q1Q2", "Q1", "Q2", "MICI", "PCT"]:
        for _, r in K[K["metric"] == m].iterrows():
            L.put(f"k.{r['method'].lower()}.{m}.iter{int(r['iteration'])}.{r['judge']}", _row(r),
                  source=f"tables/{SCRIPT}_{r['method'].lower()}_{'primary' if r['judge']==JS['primary'] else 'heldout'}.md "
                         f"(metric={m}, iteration={int(r['iteration'])}); compact: tables/{SCRIPT}_table1{'' if m=='Q1Q2' else '_'+m}.md")
    for _, r in K[(K["iteration"] == 0)].iterrows():
        L.put(f"base_vs_base.{r['method'].lower()}.{r['metric']}.{r['judge']}", _row(r),
              source=f"tables/{SCRIPT}_{r['method'].lower()}_{'primary' if r['judge']==JS['primary'] else 'heldout'}.md (iteration=0)",
              note="two independent base draws of the same 96 personas — the noise floor of the paired contrast")
    for _, r in SUM.iterrows():
        L.put(f"summary.{r['method'].lower()}.{r['metric']}.{r['judge']}",
              {k: r[k] for k in ["n_iters", "n_sig_K0_higher", "n_sig_K5_higher", "n_sig_K0_better", "n_sig_K5_better",
                                 "iters_sig_K0_higher", "iters_sig_K5_higher", "mean_delta_iters1toN", "mean_dz_iters1toN",
                                 "base_delta", "max_abs_dz", "max_abs_dz_iter", "max_abs_dz_delta"]},
              source=f"tables/{SCRIPT}_summary.md")
    # PTO Q2-vs-Q1 split: iterations where Q2 carries the K=0 edge (Q2 delta > Q1 delta and Q2 sig)
    for jk in JUDGE_KEYS:
        d = K[(K["judge"] == JS[jk]) & (K["method"] == "PTO")]
        q1 = d[d["metric"] == "Q1"].set_index("iteration"); q2 = d[d["metric"] == "Q2"].set_index("iteration")
        split = {}
        for it in sorted(set(q1.index) & set(q2.index)):
            split[f"iter{it}"] = {"Q1_delta": q1.loc[it, "mean_delta"], "Q1_dz": q1.loc[it, "dz"], "Q1_p_holm": q1.loc[it, "p_holm"],
                                  "Q2_delta": q2.loc[it, "mean_delta"], "Q2_dz": q2.loc[it, "dz"], "Q2_p_holm": q2.loc[it, "p_holm"],
                                  "Q2_carries_edge": bool(q2.loc[it, "mean_delta"] > q1.loc[it, "mean_delta"] and q2.loc[it, "p_holm"] < 0.05)}
        L.put(f"pto_q1_vs_q2_split.{JS[jk]}", split, source=f"tables/{SCRIPT}_pto_{jk}.md (metrics Q1, Q2)",
              note="Q2_carries_edge = Q2 delta exceeds Q1 delta AND Q2 Holm p<.05 (K=0 higher on Q2)")
        d = K[(K["judge"] == JS[jk]) & (K["method"] == "GRPO")]
        q1 = d[d["metric"] == "Q1"].set_index("iteration"); q2 = d[d["metric"] == "Q2"].set_index("iteration")
        L.put(f"grpo_q1_vs_q2_split_iter4_5.{JS[jk]}",
              {f"iter{it}": {"Q1_delta": q1.loc[it, "mean_delta"], "Q1_dz": q1.loc[it, "dz"], "Q1_p_holm": q1.loc[it, "p_holm"],
                             "Q2_delta": q2.loc[it, "mean_delta"], "Q2_dz": q2.loc[it, "dz"], "Q2_p_holm": q2.loc[it, "p_holm"]}
               for it in (4, 5) if it in q1.index and it in q2.index},
              source=f"tables/{SCRIPT}_grpo_{jk}.md (metrics Q1, Q2, iterations 4–5)")
    # levels
    for _, r in lvq.iterrows():
        L.put(f"level.Q1Q2.{r['arm']}.iter{int(r['iteration'])}.{r['judge']}",
              {"mean": r["mean"], "se": r["se"], "n": r["n"]}, source=f"tables/{SCRIPT}_levels.md")
    # channels (figure channels + text)
    for _, r in KC[KC["metric"].isin(FIG_CHANNELS)].iterrows():
        L.put(f"channel.{r['method'].lower()}.{r['metric']}.iter{int(r['iteration'])}.{r['judge']}", _row(r),
              source=f"tables/{SCRIPT}_channels_{r['method'].lower()}_{'primary' if r['judge']==JS['primary'] else 'heldout'}.md")
    for _, r in KT.iterrows():
        L.put(f"channel_text.{r['method'].lower()}.{r['metric']}.iter{int(r['iteration'])}", _row(r),
              source=f"tables/{SCRIPT}_channels_text_{r['method'].lower()}.md")
    for _, r in KCS.iterrows():
        L.put(f"channel_summary.{r['method'].lower()}.{r['metric']}.{r['judge']}",
              {k: r[k] for k in ["n_iters", "n_sig_K0_higher", "n_sig_K5_higher", "n_sig_K0_better", "n_sig_K5_better",
                                 "iters_sig_K0_higher", "iters_sig_K5_higher", "mean_delta_iters1toN", "max_abs_dz", "max_abs_dz_iter"]},
              source=f"tables/{SCRIPT}_channels_summary.md")
    L.put("conventions", {"sign": "+ => K=0 higher (K=0 minus K=5)", "pairing": "persona_id (n=96)",
                          "holm_family": "iterations 0..N within (judge, method, metric)",
                          "oracle_repeatability_band": ORACLE_NOISE,
                          "censoring": "GRPO_LA5 ends at iteration 5",
                          "iteration0": "two independent base draws (K=0-arm base vs K=5-arm base)"},
          source="this script")
    p = L.save(); print("ledger", p)


if __name__ == "__main__":
    main()
