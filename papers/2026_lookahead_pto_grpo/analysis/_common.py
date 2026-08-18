"""_common.py — shared plumbing for this paper's analysis generators.

Every script in this folder is a *paper-local generator*: it reads the Exp3 score lake and
training artifacts THROUGH ``eda_analysis`` (so its numbers agree with the tracked EDA tables),
and writes tables / figures / a numbers ledger into the paper folder:

    ../tables/<script>_<name>.md   (+ .csv)      readable + machine-readable
    ../figures/<script>_<name>.png (+ .pdf)      200 dpi raster + vector
    out/<script>.json                            every number the paper may quote (the NUMBERS.md source)

Run any script with the repo venv from anywhere::

    .venv/Scripts/python.exe papers/2026_lookahead_pto_grpo/analysis/<script>.py

Conventions the paper depends on
--------------------------------
* Arms are labelled ``PTO_LA0 / PTO_LA5 / GRPO_LA0 / GRPO_LA5``; a scored model state is
  ``<METHOD>Exp3_LA<K>_<Base|I<n>>``. Iteration 0 = Base (an independent draw per arm).
* K-contrast sign convention follows ``eda_analysis.stats.paired_k_comparison``:
  ``+ delta => K=0 higher``. State it in every table caption.
* Pair on ``persona_id`` (never ``file_index``) — ``load_scores`` attaches it.
* Never average the two graders' raw scores; report them side by side.
* Style: project palette (PTO cool / GRPO warm; Okabe-Ito), K=0 solid + circle, K=5 dashed +
  square; one y-axis per panel; a legend whenever >= 2 series; whitegrid; PNG + PDF.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
EXP3 = REPO / "Exp3_PTO_GRPO"
EDA_DIR = EXP3 / "eda"
RESULTS = EDA_DIR / "results"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
OUT = HERE / "out"
for _d in (TABLES, FIGURES, OUT):
    _d.mkdir(parents=True, exist_ok=True)

if str(EDA_DIR) not in sys.path:
    sys.path.insert(0, str(EDA_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import eda_analysis  # noqa: E402
from eda_analysis import EdaConfig  # noqa: E402
from eda_analysis.constants import PRIMARY_JUDGE_TAG, set_active_judge  # noqa: E402
from eda_analysis.plotting_style import arm_palette, set_style  # noqa: E402

PRIMARY = PRIMARY_JUDGE_TAG                     # "openai_gpt-4o-mini-2024-07-18"
HELDOUT = "anthropic_claude-haiku-4-5"
JUDGES = {"primary": PRIMARY, "heldout": HELDOUT}
JUDGE_SHORT = {PRIMARY: "gpt-4o-mini", HELDOUT: "claude-haiku-4-5", "": "gpt-4o-mini"}
JUDGE_LABEL = {PRIMARY: "training oracle (gpt-4o-mini)", HELDOUT: "held-out judge (Claude Haiku 4.5)"}

ARMS = ["PTO_LA0", "PTO_LA5", "GRPO_LA0", "GRPO_LA5"]
K_STYLE = {0: {"ls": "-", "marker": "o"}, 5: {"ls": "--", "marker": "s"}}
RUBRICS = ["Q1Q2", "Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"]
LOWER_IS_BETTER = {"MICI"}


def k_of(arm: str) -> int:
    return int(arm.split("_LA")[1])


def method_of(arm: str) -> str:
    return arm.split("_")[0]


def style():
    """Project publication style; call once per script before plotting."""
    set_style(EdaConfig(view="L5", context="paper", font_scale=1.0, savefig_dpi=200))


def palette(arms: Sequence[str] = ARMS) -> dict:
    return arm_palette(list(arms))


# ── loaders ──────────────────────────────────────────────────────────────────

def load_scores(judge: str = "", *, ks=(0, 5), methods=None) -> pd.DataFrame:
    """scores_long for BOTH K arms of every method under one grader.

    ``judge=""`` (or PRIMARY) = the training oracle; ``HELDOUT`` = Claude Haiku 4.5.
    Columns include: arm, iteration, model, questionnaire (metric), score, persona_id, file_index.
    Uses ``cross_k_scores`` so the arm/metric filters match the tracked EDA exactly.
    """
    tag = "" if judge in ("", PRIMARY) else judge
    cfg = EdaConfig(view="L5", judge=tag, methods=methods, verbose=False)
    set_active_judge(tag, 0)
    S = eda_analysis.notebook_setup(cfg)
    df = eda_analysis.cross_k_scores(S)
    if ks is not None:
        df = df[df["arm"].map(k_of).isin(list(ks))]
    df = df.copy()
    df["judge"] = JUDGE_SHORT[tag]
    return df


def load_scores_both() -> dict:
    """{'primary': scores_long, 'heldout': scores_long} — always call primary first, then held-out,
    and do NOT interleave with other loaders (the active judge is module-level state)."""
    out = {"primary": load_scores("")}
    out["heldout"] = load_scores(HELDOUT)
    set_active_judge("", 0)   # leave the process on the primary grader
    return out


def wide(scores_long: pd.DataFrame, metric: str) -> pd.DataFrame:
    """persona_id x model matrix for one metric (NaN where unscored)."""
    d = scores_long[scores_long["questionnaire"] == metric]
    return d.pivot_table(index="persona_id", columns="model", values="score", aggfunc="mean")


# ── paired statistics (mirror eda_analysis.stats conventions) ────────────────

def paired(a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired contrast a - b over aligned arrays (NaNs dropped pairwise). Returns mean_delta,
    dz (mean/sd of the deltas), bootstrap 95% CI, Wilcoxon p, n."""
    from scipy import stats as sps
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = int(d.size)
    if n < 3:
        return dict(n=n, mean_delta=np.nan, dz=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = float(sps.wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    return dict(n=n, mean_delta=float(d.mean()), dz=dz, ci_lo=float(lo), ci_hi=float(hi), p=p)


def holm(pvals) -> np.ndarray:
    return eda_analysis.stats.holm(list(pvals))


# ── savers ───────────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        return f"{x:.{nd}f}"
    return str(x)


def save_table(df: pd.DataFrame, name: str, *, caption: str = "", nd: int = 3,
               index: bool = False) -> Path:
    """Write ../tables/<name>.md (+ .csv). ``caption`` goes above the table (say the sign
    convention, the judge, the pairing unit, and where the numbers came from)."""
    df = df.copy()
    if index:
        df = df.reset_index()
    csv_path = TABLES / f"{name}.csv"
    md_path = TABLES / f"{name}.md"
    df.to_csv(csv_path, index=False)
    cols = list(df.columns)
    lines = []
    if caption:
        lines += [caption.strip(), ""]
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def save_fig(fig, name: str, *, formats=("png", "pdf"), dpi: int = 200) -> Path:
    """Write ../figures/<name>.png (+ .pdf) and close the figure."""
    fig.tight_layout()
    p = None
    for ext in formats:
        p = FIGURES / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return FIGURES / f"{name}.png"


class Ledger:
    """Collect every number the paper may quote into out/<script>.json.

    Usage::
        L = Ledger("k_contrast_headline")
        L.put("pto.q1q2.iter6.primary", {"delta": 0.257, "dz": 0.417, "p_holm": 0.000},
              source="tables/k_contrast_headline_pto_primary.md row iter=6")
        L.save()
    Keys are dotted paths; values are JSON-serialisable; ``source`` names the table/figure the
    number can be re-read from (the paper's NUMBERS.md cites these).
    """

    def __init__(self, script: str):
        self.script = script
        self.d: dict = {"_script": script, "numbers": {}}

    def put(self, key: str, value, *, source: str = "", note: str = ""):
        def _clean(v):
            if isinstance(v, dict):
                return {k: _clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_clean(x) for x in v]
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, float) and np.isnan(v):
                return None
            return v
        self.d["numbers"][key] = {"value": _clean(value), "source": source, "note": note}

    def save(self) -> Path:
        p = OUT / f"{self.script}.json"
        p.write_text(json.dumps(self.d, indent=1, ensure_ascii=False), encoding="utf-8")
        return p


def load_iclr_table1() -> pd.DataFrame:
    """The ICLR SSI-FM poster's Table 1 (Llama-2-7B, GPT-3.5 patient+oracle; 7 iterations),
    transcribed from papers/2025_iclr_pto_lookahead/submitted/paper.pdf. Mean scores only.
    Columns: arm (L0/L5), iteration (0 = Base), Q1, Q2, Final(=mean of Q1,Q2 means)."""
    rows = [
        ("Base", 0, 3.521, 3.385, 3.453),
        ("L0", 1, 3.863, 3.452, 3.657), ("L0", 2, 3.750, 3.435, 3.593), ("L0", 3, 3.796, 3.567, 3.682),
        ("L0", 4, 3.969, 3.585, 3.777), ("L0", 5, 3.744, 3.478, 3.611), ("L0", 6, 3.794, 3.494, 3.644),
        ("L0", 7, 3.677, 3.452, 3.565),
        ("L5", 1, 3.898, 3.523, 3.710), ("L5", 2, 3.969, 3.618, 3.794), ("L5", 3, 4.050, 3.683, 3.866),
        ("L5", 4, 3.981, 3.605, 3.793), ("L5", 5, 4.225, 3.660, 3.942), ("L5", 6, 4.112, 3.656, 3.884),
        ("L5", 7, 4.190, 3.775, 3.982),
    ]
    return pd.DataFrame(rows, columns=["arm", "iteration", "Q1", "Q2", "Final"])
