"""
data.py — load + shape the Exp3 data (discovery → personas → tidy scores → selection).

This is the "where does the data come from" layer, merged from the four former plumbing
modules so there is ONE file to open when you need to change how arms are found, how the true
persona is recovered, how ``scores_long`` is built, or how best-iteration selection works:

- **discovery**  — glob runs on disk → :class:`Arm` manifest + :func:`filter_arms` (no registry).
- **personas**   — recover the TRUE patient persona per conversation (replay the seeded shuffle).
- **scores**     — the tidy long ``scores_long`` backbone + Q1Q2 composite + subscales + derived
                   MITI-proficiency ratios + the ``select_scores`` / ``collapse_base`` / ``to_wide`` helpers.
- **selection**  — :func:`all_models` vs :func:`best_per_experiment` (peak iter by own oracle).

Read-only, disk-discovery-driven. Public names are re-exported from ``eda_analysis/__init__.py``;
the legacy submodule names (``scores``/``discovery``/``personas``/``select``) are aliased to this
module there too, so older ``eda_analysis.personas.canonical_personas`` references still resolve.
"""

import glob
import json
import os
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import DATA_DIR, PERSONA_COLS, QUESTIONNAIRES


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DISCOVERY — find Exp3 runs on disk and describe them (no registry)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Globs ``data/{pto,grpo}_Exp3/conversations/full/<EXP_NAME>/model_iter_<k>_TT*_TP*`` and reads
# the sibling ``runs/full/<EXP_NAME>/run_metadata.json`` for the seed + training config. One
# :class:`Arm` per run. Experiment-name schemes (see Exp3 CLAUDE.md):
#   GRPO: GRPO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_G{G}
#   PTO:  PTO_Iterative_{Oracle}_Llama32-1B_LA{K}_MCL{MCL}_M{M}_PT{greedy|indep}

_METHOD_DIRS = {"PTO": "pto_Exp3", "GRPO": "grpo_Exp3"}
_MODEL_PREFIX = {"PTO": "PTOExp3", "GRPO": "GRPOExp3"}
_DEFAULT_SEED = 42  # all current runs; only used if run_metadata.json is missing
_ITER_RE = re.compile(r"model_iter_(\d+)_")

_EXP_RE = re.compile(
    r"^(?P<method>PTO|GRPO)_Iterative_(?P<oracle>[A-Za-z0-9]+)_Llama32-1B_"
    r"LA(?P<K>\d+)_MCL(?P<mcl>\d+)_(?:G(?P<g>\d+)|M(?P<m>\d+)_PT(?P<mode>greedy|indep))$"
)


def parse_experiment_name(exp_name: str) -> Optional[dict]:
    """Parse an EXPERIMENT_NAME folder into its fields, or ``None`` if it doesn't match."""
    m = _EXP_RE.match(exp_name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "method": d["method"],
        "oracle": d["oracle"],
        "K": int(d["K"]),
        "mcl": int(d["mcl"]),
        "mode": d["mode"] or ("group" if d["method"] == "GRPO" else None),
        "branches": int(d["g"]) if d["g"] else (int(d["m"]) if d["m"] else None),
    }


@dataclass
class Arm:
    """One discovered training run (a method × K × MCL × mode arm)."""
    method: str                 # "PTO" | "GRPO"
    exp_name: str
    K: int
    mcl: int
    mode: Optional[str]
    oracle: str                 # training-oracle token, e.g. "Q1Q2"
    seed: int
    n_personas: int
    conv_dirs: Dict[int, str]   # model_iter k -> abs conversation dir
    runs_dir: str
    eval_root: str
    config: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.method}_LA{self.K}"

    @property
    def iters(self) -> List[int]:
        return sorted(self.conv_dirs)

    def model_name(self, k: int) -> str:
        prefix = _MODEL_PREFIX[self.method]
        tail = "Base" if k == 0 else f"I{k}"
        return f"{prefix}_LA{self.K}_{tail}"

    def eval_oracle_label(self, k: int) -> str:
        return "none" if k == 0 else self.oracle

    def eval_dir(self, k: int, metric_subdir: str) -> str:
        return os.path.join(
            self.eval_root, f"metric={metric_subdir}",
            f"oracle={self.eval_oracle_label(k)}", self.model_name(k),
        )

    def conv_dir(self, k: int) -> Optional[str]:
        return self.conv_dirs.get(k)


def discover_arms(data_dir: str = DATA_DIR, *, include_archived: bool = False) -> List[Arm]:
    """Discover all Exp3 arms present on disk, newest-data-first within method."""
    arms: List[Arm] = []
    for method, mdir in _METHOD_DIRS.items():
        conv_root = os.path.join(data_dir, mdir, "conversations", "full")
        if not os.path.isdir(conv_root):
            continue
        for exp_name in sorted(os.listdir(conv_root)):
            if not include_archived and "Archive" in exp_name:
                continue
            exp_path = os.path.join(conv_root, exp_name)
            if not os.path.isdir(exp_path):
                continue
            parsed = parse_experiment_name(exp_name)
            if parsed is None:
                continue
            # iter dirs present
            conv_dirs: Dict[int, str] = {}
            for d in glob.glob(os.path.join(exp_path, "model_iter_*")):
                m = _ITER_RE.search(os.path.basename(d) + "_")
                if m and os.path.isdir(d):
                    conv_dirs[int(m.group(1))] = d
            if not conv_dirs:
                continue
            runs_dir = os.path.join(data_dir, mdir, "runs", "full", exp_name)
            seed, cfg = _read_seed_config(runs_dir)
            arms.append(Arm(
                method=parsed["method"], exp_name=exp_name, K=parsed["K"],
                mcl=parsed["mcl"], mode=parsed["mode"], oracle=parsed["oracle"],
                seed=seed, n_personas=int(cfg.get("num_conversations_per_iter", 96)),
                conv_dirs=conv_dirs, runs_dir=runs_dir,
                eval_root=os.path.join(data_dir, mdir, "eval_scores"), config=cfg,
            ))
    return arms


def filter_arms(arms: List[Arm], *, methods=None, ks=None, modes=None,
                arm_labels=None) -> List[Arm]:
    """Filter a discovered arm list by method / K / mode / explicit label (each None = no filter).

    Used by ``notebook_setup`` to honour ``EdaConfig`` arm selection (and the VIEW knob, which
    drives ``ks``). ``arm_labels`` is an explicit whitelist on ``Arm.label`` (e.g. ``["PTO_LA0"]``),
    applied alongside the field filters.
    """
    def keep(a: Arm) -> bool:
        if methods and a.method not in set(methods):
            return False
        if ks is not None and a.K not in set(ks):
            return False
        if modes and a.mode not in set(modes):
            return False
        if arm_labels and a.label not in set(arm_labels):
            return False
        return True
    return [a for a in arms if keep(a)]


def _read_seed_config(runs_dir: str):
    """Return ``(seed, config_dict)`` from ``run_metadata.json`` (defaults if absent)."""
    meta = os.path.join(runs_dir, "run_metadata.json")
    if os.path.exists(meta):
        try:
            cfg = json.load(open(meta)).get("config", {})
            return int(cfg.get("seed", _DEFAULT_SEED)), cfg
        except Exception:
            pass
    print(f"  [discovery] WARNING: no run_metadata.json under {runs_dir} — "
          f"assuming seed={_DEFAULT_SEED}")
    return _DEFAULT_SEED, {}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PERSONAS — recover the TRUE patient persona behind each saved conversation    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# The trainer simulates the same 96 personas every iteration but in a *seeded shuffled order*,
# saving each under its shuffled position (``conversation_{position}.csv``). So
# ``conversation_{i}.csv`` is a different persona each iteration. The shuffle is deterministic:
#   iter_rng = random.Random(cfg.seed + iteration)          # in-loop, saves model_iter_{iteration-1}
#   final    = random.Random(cfg.seed + num_iterations + 1) # saves model_iter_{N}
# which collapses to: model_iter_k uses shuffle seed ``seed + k + 1``. Replaying
# ``Random(seed+k+1).shuffle(list(range(96)))`` reproduces ``order`` where
# ``order[file_index] = canonical_persona_id``. Validated against turn-1 age/gender — exact.

@lru_cache(maxsize=1)
def canonical_personas(n: int = 96) -> pd.DataFrame:
    """The 96 canonical personas + characteristics, indexed by ``persona_id`` 0..n-1.

    Columns: :data:`PERSONA_COLS` (gender, age_value, problem, problem_time,
    tried_to_solve, cooperation_level).
    """
    from system_prompts_builder import get_patient_permutation_characteristics
    rows = []
    for pid in range(n):
        ch = get_patient_permutation_characteristics(pid) or {}
        row = {"persona_id": pid}
        row.update({c: ch.get(c) for c in PERSONA_COLS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("persona_id")


def persona_order(seed: int, model_iter: int, n: int = 96) -> List[int]:
    """``order`` where ``order[file_index] == canonical_persona_id`` for ``model_iter``.

    Reproduces the trainer's ``shuffled = list(all_permutations); rng.shuffle(shuffled)``
    by replaying the identical seeded shuffle on ``list(range(n))`` (the swap sequence is
    content-independent, so this is exact).
    """
    order = list(range(n))
    random.Random(seed + model_iter + 1).shuffle(order)
    return order


def file_to_persona(seed: int, model_iter: int, n: int = 96) -> Dict[int, int]:
    """``{file_index: canonical_persona_id}`` for ``model_iter``."""
    return {i: pid for i, pid in enumerate(persona_order(seed, model_iter, n))}


def attach_personas(
    df: pd.DataFrame,
    seed: int,
    *,
    iter_col: str = "iteration",
    file_col: str = "file_index",
    n: int = 96,
) -> pd.DataFrame:
    """Add ``persona_id`` + :data:`PERSONA_COLS` to *df* by replaying the per-iter shuffle.

    *df* must carry an iteration column (``model_iter`` k) and a file-index column (the saved
    ``conversation_{i}`` / ``{patient_id}.csv`` index). One ``seed`` per call — split by arm
    before calling if arms differ in seed (they don't today).
    """
    out = df.copy()
    cano = canonical_personas(n)
    # Build the (iter, file_index) -> persona_id map only for the iters present.
    pid = []
    cache: Dict[int, List[int]] = {}
    for it, fi in zip(out[iter_col].astype(int), out[file_col].astype(int)):
        order = cache.get(it)
        if order is None:
            order = persona_order(seed, it, n)
            cache[it] = order
        pid.append(order[fi] if 0 <= fi < n else -1)
    out["persona_id"] = pid
    chars = cano.reindex(out["persona_id"].values).reset_index(drop=True)
    for c in PERSONA_COLS:
        out[c] = chars[c].values
    return out


def validate_recovery(
    conv_dir_for_iter,
    seed: int,
    iters: List[int],
    *,
    n: int = 96,
    sample_every: int = 8,
    verbose: bool = True,
) -> dict:
    """Assert the replay is sound and (optionally) matches conversation content.

    ``conv_dir_for_iter(k)`` -> absolute path to ``model_iter_k``'s conversation folder (so this
    stays IO-agnostic). Checks per iter that recovered ids form a full 0..n-1 permutation, and
    that the age stated in the patient's first turn matches the recovered persona on a sampled
    subset. Returns a small report; raises ``AssertionError`` if the permutation check fails.
    """
    cano = canonical_personas(n)
    age_ok = age_tot = 0
    for k in iters:
        order = persona_order(seed, k, n)
        assert sorted(order) == list(range(n)), (
            f"persona recovery for model_iter_{k} is not a permutation of 0..{n-1} "
            f"(seed={seed}); the seed/order assumption is wrong."
        )
        cdir = conv_dir_for_iter(k)
        if not cdir or not os.path.isdir(cdir):
            continue
        for fi in range(0, n, sample_every):
            fp = os.path.join(cdir, f"conversation_{fi}.csv")
            if not os.path.exists(fp):
                continue
            try:
                cdf = pd.read_csv(fp)
            except Exception:
                continue
            pt = cdf[cdf["role"] == "patient"]["conversation"]
            if not len(pt):
                continue
            m = re.search(r"\b(\d{2})\b", str(pt.iloc[0])[:120])
            if not m:
                continue
            age_tot += 1
            if str(cano.loc[order[fi], "age_value"]) == m.group(1):
                age_ok += 1
    rep = {
        "iters_checked": list(iters),
        "permutation_ok": True,
        "age_match": (age_ok, age_tot),
        "age_match_rate": (age_ok / age_tot) if age_tot else None,
    }
    if verbose:
        rate = rep["age_match_rate"]
        print(f"persona recovery: permutation OK for iters {list(iters)}; "
              f"age-in-intro match {age_ok}/{age_tot}"
              + (f" ({rate:.0%})" if rate is not None else ""))
    return rep


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SCORES — the tidy long backbone every analysis derives from                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ``load_scores_long`` reads each arm's per-conversation eval CSVs, recovers the true persona,
# and returns one row per (arm, iteration, persona, questionnaire) -> score.
# Composite: ``Q1Q2 = mean(Q1_Mean, Q2_Mean)`` (a [1,5] mean, matching the headline axis).

# Display name -> per-conv mean column (the non-composite rubrics).
MEAN_COLS = {disp: meancol for disp, (sub, meancol) in QUESTIONNAIRES.items() if sub is not None}

_KEY = ["method", "arm", "K", "mcl", "mode", "oracle", "model", "iteration", "is_base", "file_index"]


def load_scores_long(arms: Optional[List] = None, *, attach_persona: bool = True) -> pd.DataFrame:
    """Tidy long eval scores across all discovered arms.

    Columns: ``method, arm, K, mcl, mode, oracle, model, iteration, is_base, file_index,
    questionnaire, score`` (+ ``persona_id`` & characteristics if ``attach_persona``). Includes
    the ``Q1Q2`` composite. Missing eval folders are skipped, so partially-scored arms contribute
    whatever exists.
    """
    arms = discover_arms() if arms is None else arms
    rows = []
    for arm in arms:
        for k in arm.iters:
            base_meta = {
                "method": arm.method, "arm": arm.label, "K": arm.K, "mcl": arm.mcl,
                "mode": arm.mode, "oracle": arm.oracle, "model": arm.model_name(k),
                "iteration": k, "is_base": (k == 0),
            }
            for disp, (sub, meancol) in QUESTIONNAIRES.items():
                if sub is None:  # composite — built after the raw load
                    continue
                ddir = arm.eval_dir(k, sub)
                if not os.path.isdir(ddir):
                    continue
                for fn in os.listdir(ddir):
                    stem, ext = os.path.splitext(fn)
                    if ext != ".csv" or not stem.isdigit():
                        continue
                    try:
                        r = pd.read_csv(os.path.join(ddir, fn))
                    except Exception:
                        continue
                    if len(r) == 0 or meancol not in r.columns:
                        continue
                    rows.append({**base_meta, "file_index": int(stem),
                                 "questionnaire": disp, "score": float(r.iloc[0][meancol])})
    long = pd.DataFrame(rows)
    if long.empty:
        return long

    long = _add_q1q2_composite(long)
    if attach_persona:
        # seed is constant per arm; current runs all share seed — attach per arm.
        seed_by_arm = {a.label: a.seed for a in arms}
        parts = []
        for arm_label, g in long.groupby("arm", sort=False):
            parts.append(attach_personas(g, seed_by_arm.get(arm_label, 42)))
        long = pd.concat(parts, ignore_index=True)
    return long


def _add_q1q2_composite(long: pd.DataFrame) -> pd.DataFrame:
    """Append the ``Q1Q2`` = mean(Q1, Q2) composite rows (where both components exist)."""
    comp_src = long[long["questionnaire"].isin(["Q1", "Q2"])]
    if comp_src.empty:
        return long
    wide = comp_src.pivot_table(index=_KEY, columns="questionnaire", values="score")
    if not {"Q1", "Q2"}.issubset(wide.columns):
        return long
    wide = wide.dropna(subset=["Q1", "Q2"])
    comp = wide.reset_index()
    comp["questionnaire"] = "Q1Q2"
    comp["score"] = comp[["Q1", "Q2"]].mean(axis=1)
    comp = comp[_KEY + ["questionnaire", "score"]]
    return pd.concat([long, comp], ignore_index=True)


_SUBSCALES = {
    "WAI-SR": ("WAI_SR", {"WAI_Goal_Mean": "Goal", "WAI_Task_Mean": "Task", "WAI_Bond_Mean": "Bond"}),
    "MITI": ("MITI", {"MITI1_CultivatingChangeTalk": "ChangeTalk", "MITI2_SofteningSustainTalk": "SoftenSustain",
                       "MITI3_Partnership": "Partnership", "MITI4_Empathy": "Empathy"}),
}


def load_subscales(arms: Optional[List] = None) -> pd.DataFrame:
    """Tidy long frame of WAI (Goal/Task/Bond) + MITI (4 globals) subscales.

    One row per (arm, iteration, file_index, parent questionnaire, subscale) -> score.
    Used by the familiar 'subscales' view; complements the headline-mean `scores_long`.
    """
    arms = discover_arms() if arms is None else arms
    rows = []
    for arm in arms:
        for k in arm.iters:
            for parent, (sub, cols) in _SUBSCALES.items():
                ddir = arm.eval_dir(k, sub)
                if not os.path.isdir(ddir):
                    continue
                for fn in os.listdir(ddir):
                    stem, ext = os.path.splitext(fn)
                    if ext != ".csv" or not stem.isdigit():
                        continue
                    try:
                        r = pd.read_csv(os.path.join(ddir, fn)).iloc[0]
                    except Exception:
                        continue
                    for src, name in cols.items():
                        if src in r.index and pd.notna(r[src]):
                            rows.append({"arm": arm.label, "method": arm.method, "K": arm.K,
                                         "model": arm.model_name(k), "iteration": k,
                                         "is_base": (k == 0), "file_index": int(stem),
                                         "parent": parent, "subscale": name, "score": float(r[src])})
    return pd.DataFrame(rows)


def select_scores(scores_long: pd.DataFrame, *, arms: Optional[List] = None,
                  iters: Optional[List] = None, metrics: Optional[List] = None) -> pd.DataFrame:
    """Slice ``scores_long`` to chosen arms / iterations / metrics (each None = keep all).

    The one selection helper every figure cell uses, so a notebook can point a plot at a subset
    (e.g. ``select_scores(S.SCORES, arms=["PTO_LA0","GRPO_LA0"])``) instead of looping per arm.
    """
    d = scores_long
    if arms is not None:
        d = d[d["arm"].isin(list(arms))]
    if iters is not None:
        d = d[d["iteration"].isin(list(iters))]
    if metrics is not None:
        d = d[d["questionnaire"].isin(list(metrics))]
    return d


def collapse_base(scores_long: pd.DataFrame, *, label: str = "Base") -> pd.DataFrame:
    """Pool every arm's iter-0 base into ONE descriptive model row block.

    All arms share the same base policy (frozen Llama-3.2-1B) on the same iter-0 persona order
    (shuffle ``seed+1`` for every arm), so the per-arm ``*_Base`` rows are near-replicates. For
    cross-model *descriptive* views (bars / subscales) this relabels them to a single pooled
    model — decluttering the axis and giving a higher-N base reference.

    Relabel: ``model=label``, ``arm=label``, ``method="Base"``, ``K=-1`` (so ``plotting.model_order``
    sorts it first). Non-base rows pass through untouched.

    NOTE: descriptive only — do **not** feed this to the persona-paired / vs-base ``stats.*``
    helpers, which must keep pairing each arm against its OWN base.
    """
    if scores_long.empty or "is_base" not in scores_long.columns:
        return scores_long
    out = scores_long.copy()
    base = out["is_base"]
    out.loc[base, "model"] = label
    out.loc[base, "arm"] = label
    if "method" in out.columns:
        out.loc[base, "method"] = "Base"
    if "K" in out.columns:
        out.loc[base, "K"] = -1
    return out


def add_derived_mitiprof_rows(scores_long: pd.DataFrame,
                              arms: Optional[List] = None) -> pd.DataFrame:
    """Append the **objective MITI-proficiency ratios** as extra ``questionnaire`` rows.

    Derived for FREE from the already-scored MITI behavior counts (no oracle re-run):

    - ``R:Q``   = (SR + CR) / Q                      — reflection-to-question ratio
    - ``%CR``   = CR / (SR + CR)                      — proportion complex reflections
    - ``%MICO`` = (SR+CR+AF+Seek) / (SR+CR+AF+Seek+Persuade)  — MI-consistent proportion

    These ratios are technique metrics (not warmth halos), so they belong in the inter-rubric
    correlation/PCA as candidate *orthogonal* axes. Rows are aligned to the existing
    ``scores_long`` conversation identities by (arm, iteration, file_index), inheriting the full
    key + persona columns, so they pivot onto the same rows in :func:`to_wide`.
    Returns ``scores_long`` unchanged if MITI behavior data is unavailable.
    """
    from .behavior import load_miti_behavior  # deferred: behavior is a separate (kept) module
    if scores_long.empty:
        return scores_long
    # Idempotent: notebook_setup already appends these, so a notebook re-calling is a no-op.
    if "R:Q" in set(scores_long["questionnaire"].unique()):
        return scores_long
    miti = load_miti_behavior(arms, attach_persona=False)
    if miti.empty:
        return scores_long

    def _ratio(num, den):
        return num / den if (den is not None and den > 0) else None

    recs = []
    for _, r in miti.iterrows():
        sr, cr = r.get("B4_SR") or 0, r.get("B5_CR") or 0
        q, af, seek = r.get("B3_Q") or 0, r.get("B6_AF") or 0, r.get("B7_Seek") or 0
        pers = r.get("B2_Persuade") or 0
        mico = sr + cr + af + seek
        recs.append({"arm": r["arm"], "iteration": r["iteration"], "file_index": r["file_index"],
                     "R:Q": _ratio(sr + cr, q), "%CR": _ratio(cr, sr + cr),
                     "%MICO": _ratio(mico, mico + pers)})
    deriv = pd.DataFrame(recs)
    if deriv.empty:
        return scores_long

    # Skeleton of conversation identities (full key + persona) from scores_long.
    id_cols = [c for c in scores_long.columns if c not in ("questionnaire", "score")]
    skel = scores_long[id_cols].drop_duplicates(["arm", "iteration", "file_index"])
    merged = skel.merge(deriv, on=["arm", "iteration", "file_index"], how="inner")
    if merged.empty:
        return scores_long
    long_new = merged.melt(id_vars=id_cols, value_vars=["R:Q", "%CR", "%MICO"],
                           var_name="questionnaire", value_name="score").dropna(subset=["score"])
    return pd.concat([scores_long, long_new], ignore_index=True)


def to_wide(scores_long: pd.DataFrame, value: str = "score") -> pd.DataFrame:
    """Pivot to one row per (arm, iteration, persona) with a column per questionnaire.

    Convenient for paired stats + inter-rubric correlation. Persona characteristics are carried
    through if present.
    """
    idx = ["method", "arm", "K", "oracle", "model", "iteration", "is_base", "file_index"]
    if "persona_id" in scores_long.columns:
        idx.append("persona_id")
    wide = scores_long.pivot_table(index=idx, columns="questionnaire", values=value).reset_index()
    if "persona_id" in scores_long.columns:
        chars = (scores_long[["persona_id"] + [c for c in PERSONA_COLS if c in scores_long.columns]]
                 .drop_duplicates("persona_id"))
        wide = wide.merge(chars, on="persona_id", how="left")
    wide.columns.name = None
    return wide


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SELECTION — all iterations vs best-per-arm (by own training oracle)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Training-oracle token -> the questionnaire display name that judges it.
_OWN_ORACLE = {
    "Q1Q2": "Q1Q2", "WAI": "WAI-SR", "CSQ8": "CSQ-8", "MI_SAT": "MI-SAT", "MITI": "MITI",
}


def all_models(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Identity passthrough (every iteration of every arm)."""
    return scores_long


def best_per_experiment(
    scores_long: pd.DataFrame,
    by: str = "own_oracle",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep each arm's base + its best-scoring iteration on its own oracle.

    Returns ``(filtered_scores_long, summary)`` where ``summary`` has one row per arm: the
    selected best iteration, its own-oracle mean, and n personas. Ties break to the earliest
    iteration.
    """
    if by != "own_oracle":
        raise ValueError(f"unsupported selection mode {by!r} (only 'own_oracle')")
    if scores_long.empty:
        return scores_long, pd.DataFrame()

    keep_models, summary_rows = [], []
    for (arm, oracle), g in scores_long.groupby(["arm", "oracle"], sort=False):
        judge = _OWN_ORACLE.get(oracle)
        sub = g[g["questionnaire"] == judge] if judge else g.iloc[0:0]
        # always keep the base
        base_models = g.loc[g["is_base"], "model"].unique().tolist()
        keep_models += base_models
        if sub.empty:
            continue
        per_iter = (sub[~sub["is_base"]]
                    .groupby(["iteration", "model"], observed=True)["score"]
                    .mean().reset_index()
                    .sort_values(["score", "iteration"], ascending=[False, True]))
        if per_iter.empty:
            continue
        best = per_iter.iloc[0]
        keep_models.append(best["model"])
        summary_rows.append({
            "arm": arm, "oracle": oracle, "judged_by": judge,
            "best_iteration": int(best["iteration"]), "best_model": best["model"],
            "own_oracle_mean": round(float(best["score"]), 4),
            "n": int((sub["iteration"] == best["iteration"]).sum()),
        })

    filtered = scores_long[scores_long["model"].isin(set(keep_models))].copy()
    summary = pd.DataFrame(summary_rows).sort_values("arm").reset_index(drop=True)
    return filtered, summary
