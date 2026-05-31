"""
data.py — Loading and shaping the EDA data.

Three concerns, in order:
1. **Conversations** — read per-model CSVs into a single DataFrame, attach
   patient characteristics, parse model-name metadata.
2. **Evaluation** — read per-questionnaire score CSVs, merge Q1+Q2 composite,
   build the ``test_cases`` list that drives every downstream analysis.
3. **Selection** — pick the best iteration per experiment group (the only
   selection strategy the EDA exposes) and filter helpers.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from system_prompts_builder import get_patient_permutation_characteristics

from .config import (
    COMPOSITE_METRICS,
    DPO_GROUP_ORDER,
    EXPERIMENT_PALETTE,
    GROUP_ORDER,
    ORACLE_METRIC_MAP,
    ORACLE_ORDER,
    ORACLE_TOKEN_ALIASES,
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       CONVERSATION LOADING                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _preprocess_conversation(conversation_df: pd.DataFrame, conversation_id: int = -1) -> dict:
    """Normalize legacy column-name typos and return a dict for one conversation."""
    df = conversation_df.rename(columns={
        "session_endded_by": "session_ended_by",
        "session_endded_explanation": "session_ended_explanation",
    })
    utterances = df["conversation"].tolist()
    return {
        "id": conversation_id,
        "conversation_length": len(utterances),
        "session_ended_by": df["session_ended_by"].iloc[0],
        "session_ended_explanation": df["session_ended_explanation"].iloc[0],
        "conversation": utterances,
    }


def _load_one(data_path: str, start_idx: int = 0, end_idx: int = 96) -> pd.DataFrame:
    """Load up to 96 ``conversation_{i}.csv`` files from one model directory."""
    conversations = []
    for i in range(start_idx, end_idx):
        path = os.path.join(data_path, f"conversation_{i}.csv")
        try:
            df = pd.read_csv(path)
            conversations.append(_preprocess_conversation(df, i))
        except FileNotFoundError:
            pass
    return pd.DataFrame(conversations)


def load_data(data_paths: List[str]) -> List[pd.DataFrame]:
    """Load every directory in *data_paths*; one DataFrame per path."""
    return [_load_one(p) for p in data_paths]


def combine_data(data_sets: List[pd.DataFrame], model_names: List[str]) -> pd.DataFrame:
    """Concatenate per-model DataFrames, adding a ``Model`` column to each."""
    return pd.concat(
        [d.assign(Model=name) for d, name in zip(data_sets, model_names)],
        ignore_index=True,
    )


def reconstruct_conversation_text(utterances: List[str]) -> str:
    """Convert an utterance list into a ``[THERAPIST]/[PATIENT]``-labeled transcript."""
    return "\n".join(
        f"[{'THERAPIST' if i % 2 == 0 else 'PATIENT'}]: {utt}"
        for i, utt in enumerate(utterances)
    )


def add_patient_characteristics(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Join patient demographic columns by ``id_col`` (looks up permutation index)."""
    def _get(pid):
        try:
            return get_patient_permutation_characteristics(int(pid))
        except Exception:
            return {}
    chars_df = pd.json_normalize(df[id_col].apply(_get))
    expected = ["gender", "age_value", "problem", "problem_time", "tried_to_solve", "cooperation_level"]
    for col in expected:
        if col not in chars_df.columns:
            chars_df[col] = np.nan
    return pd.concat(
        [df.reset_index(drop=True), chars_df[expected].reset_index(drop=True)],
        axis=1,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       MODEL METADATA PARSING                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


_MODEL_NAME_PATTERN  = re.compile(r"^L(?P<lookahead>\d+)_(?P<oracle>[A-Za-z0-9_]+)_V(?P<iteration>\d+)$")
_GRPOExp3_NAME_PATTERN = re.compile(r"^GRPOExp3_(?:(?P<base>Base)|I(?P<iter>\d+))$")
_PTOExp3_NAME_PATTERN  = re.compile(r"^PTOExp3_(?:(?P<base>Base)|I(?P<iter>\d+))$")


def _normalize_oracle_token(token: str, *, strict: bool = False) -> str:
    """Canonicalize oracle name tokens to keys used in ORACLE_METRIC_MAP.

    Aliases live in ``config.ORACLE_TOKEN_ALIASES``. Unknown tokens fall through
    to the original token (and downstream show up in the ``"Other"`` group);
    pass ``strict=True`` to raise ``ValueError`` instead — useful when adding
    new model-name conventions.
    """
    normalized = str(token).upper().replace("-", "_")
    if normalized in ORACLE_TOKEN_ALIASES:
        return ORACLE_TOKEN_ALIASES[normalized]
    if strict:
        raise ValueError(
            f"Unknown oracle token: {token!r} (normalized={normalized!r}). "
            f"Add it to config.ORACLE_TOKEN_ALIASES."
        )
    return token


def parse_model_metadata(model_name: str) -> Dict[str, Any]:
    """Parse a model name into ``{LookAhead, OracleGroup, Iteration, ExperimentGroup}``."""
    model_name = str(model_name)
    if model_name == "Base":
        return {"LookAhead": -1, "OracleGroup": "Base", "Iteration": np.nan, "ExperimentGroup": "Base"}

    m = _GRPOExp3_NAME_PATTERN.match(model_name)
    if m:
        is_base = bool(m.group("base"))
        iteration = 0 if is_base else int(m.group("iter"))
        oracle = "Base" if is_base else "Q1Q2"
        return {"LookAhead": -1, "OracleGroup": oracle, "Iteration": iteration, "ExperimentGroup": "GRPO_Exp3"}

    m = _PTOExp3_NAME_PATTERN.match(model_name)
    if m:
        is_base = bool(m.group("base"))
        iteration = 0 if is_base else int(m.group("iter"))
        oracle = "Base" if is_base else "Q1Q2"
        return {"LookAhead": -1, "OracleGroup": oracle, "Iteration": iteration, "ExperimentGroup": "PTO_Exp3"}

    m = _MODEL_NAME_PATTERN.match(model_name)
    if m:
        la = int(m.group("lookahead"))
        og = _normalize_oracle_token(m.group("oracle"))
        it = int(m.group("iteration"))
        return {"LookAhead": la, "OracleGroup": og, "Iteration": it, "ExperimentGroup": f"L{la}_{og}"}

    return {"LookAhead": -1, "OracleGroup": "Other", "Iteration": np.nan, "ExperimentGroup": model_name}


def add_model_metadata_columns(df: pd.DataFrame, model_col: str = "Model") -> pd.DataFrame:
    """Add LookAhead / OracleGroup / Iteration / ExperimentGroup / ModelGroup columns."""
    if "ExperimentGroup" in df.columns or model_col not in df.columns:
        return df
    meta = df[model_col].astype(str).apply(parse_model_metadata).apply(pd.Series)
    for col in ["LookAhead", "OracleGroup", "Iteration", "ExperimentGroup"]:
        df[col] = meta[col].values
    df["ModelGroup"] = df["OracleGroup"]
    return df


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       ORDERING & PALETTES                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


_NON_DPO_GROUPS = ("Base", "GRPO_Exp3", "PTO_Exp3")
_LOOKAHEAD_RANK = {0: 0, 5: 1}  # known look-ahead values rank ahead of "other"


def _group_rank(experiment_group: str) -> int:
    """Where this experiment group sits relative to Base / DPO / GRPO_Exp3."""
    return GROUP_ORDER.get(experiment_group, DPO_GROUP_ORDER)


def _oracle_rank(oracle: str) -> int:
    """Position of *oracle* in ``ORACLE_ORDER``, or one past the end for unknowns."""
    try:
        return ORACLE_ORDER.index(oracle)
    except ValueError:
        return len(ORACLE_ORDER)


def _model_sort_key(model_name):
    md = parse_model_metadata(model_name)
    group_rank = _group_rank(md["ExperimentGroup"])
    if md["ExperimentGroup"] in _NON_DPO_GROUPS:
        lookahead_rank = -1
    else:
        lookahead_rank = _LOOKAHEAD_RANK.get(int(md["LookAhead"]), 2)
    oracle = str(md["OracleGroup"])
    # GRPO_Exp3 / PTO_Exp3 don't sweep oracle; rank 0 keeps them ahead of the unknown bucket.
    oracle_rank = 0 if md["ExperimentGroup"] in ("GRPO_Exp3", "PTO_Exp3") else _oracle_rank(oracle)
    it = md["Iteration"]
    iter_rank = int(it) if not pd.isna(it) else 10**9
    return (group_rank, lookahead_rank, oracle_rank, oracle, iter_rank, str(model_name))


def compute_model_order(models):
    """Return a stable model order using the project's sort rules."""
    return sorted({str(m) for m in models}, key=_model_sort_key)


def apply_model_order(df, model_order=None, model_col: str = "Model"):
    """Set ``df[model_col]`` to a Categorical with the given order (computed if None)."""
    if model_col not in df.columns:
        return df
    if model_order is None:
        model_order = compute_model_order(df[model_col].dropna().astype(str).tolist())
    out = df.copy()
    out[model_col] = pd.Categorical(out[model_col], categories=model_order, ordered=True)
    return out


def _experiment_sort_key(experiment_group):
    group_rank = _group_rank(experiment_group)
    if experiment_group in _NON_DPO_GROUPS:
        return (group_rank, -1, -1, str(experiment_group))
    m = re.match(r"^L(?P<la>\d+)_(?P<oracle>[A-Za-z0-9]+)$", str(experiment_group))
    if not m:
        return (group_rank, 99, len(ORACLE_ORDER), str(experiment_group))
    la_rank = _LOOKAHEAD_RANK.get(int(m.group("la")), 2)
    return (group_rank, la_rank, _oracle_rank(m.group("oracle")), str(experiment_group))


def build_experiment_palette(df: pd.DataFrame) -> dict:
    """Return a palette dict keyed by ExperimentGroup, with fallback colors for unknowns."""
    meta = add_model_metadata_columns(df.copy()) if "ExperimentGroup" not in df.columns else df
    groups = sorted(
        meta["ExperimentGroup"].dropna().astype(str).unique().tolist(),
        key=_experiment_sort_key,
    )
    palette = {g: c for g, c in EXPERIMENT_PALETTE.items() if g in groups}
    missing = [g for g in groups if g not in palette]
    if missing:
        fallback = sns.color_palette("tab10", n_colors=len(missing)).as_hex()
        palette.update(dict(zip(missing, fallback)))
    return palette


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         EVALUATION LOADING                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _load_eval_for_models(
    model_folders: Dict[str, str],
    add_characteristics: bool = True,
) -> Optional[pd.DataFrame]:
    """Load per-patient eval CSVs given a ``{model: folder}`` map.

    Layout: ``folder/{patient_id}.csv``. Each model points at its own folder, so
    models from different methods (living under different ``eval_scores`` roots)
    are concatenated transparently. Returns ``None`` when no matching files exist.
    """
    rows = []
    for model, mf in model_folders.items():
        if not os.path.isdir(mf):
            continue
        for fn in sorted(
            os.listdir(mf),
            key=lambda fn: int(os.path.splitext(fn)[0]) if os.path.splitext(fn)[0].isdigit() else 0,
        ):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(mf, fn)
            pid = int(os.path.splitext(fn)[0]) if os.path.splitext(fn)[0].isdigit() else -1
            r = pd.read_csv(fp)
            r["Model"] = model
            r["patient_id"] = pid
            rows.append(r)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df = add_model_metadata_columns(df)
    if add_characteristics:
        df = add_patient_characteristics(df, id_col="patient_id")
    return df


def merge_q1_q2_results(q1_df: Optional[pd.DataFrame], q2_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Inner-join Q1 + Q2 on (Model, patient_id); compute the Q1Q2 composite.

    The composite (Q1Q2_Mean) is defined in ``config.COMPOSITE_METRICS`` so
    future composites (e.g. MITI_GlobalMean) can be added there. ``Q1Q2_Total``
    is computed opportunistically when the source totals are present.
    """
    if q1_df is None or q2_df is None:
        return None
    q2_cols = ["Model", "patient_id"] + [c for c in q2_df.columns if c.startswith("Q2_")]
    merged = pd.merge(q1_df.copy(), q2_df[q2_cols].copy(), on=["Model", "patient_id"], how="inner")
    if merged.empty:
        return None

    # Apply each composite that has all its source columns present.
    for out_col, spec in COMPOSITE_METRICS.items():
        sources = spec["sources"]
        if not all(c in merged.columns for c in sources):
            continue
        if spec.get("aggregator", "mean") == "mean":
            merged[out_col] = merged[sources].mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregator {spec.get('aggregator')!r} for {out_col}")

    if "Q1_Total" in merged.columns and "Q2_Total" in merged.columns:
        merged["Q1Q2_Total"] = merged["Q1_Total"] + merged["Q2_Total"]
    return add_model_metadata_columns(merged)


def load_all_eval_results(
    model_layout: Dict[str, Dict[str, str]],
    model_list: List[str],
    model_order: Optional[list] = None,
    questionnaire_dirs: Optional[Dict[str, str]] = None,
) -> dict:
    """Load eval scores for *model_list*, each model read from its labelled folder.

    ``model_layout`` maps model name -> ``{'root', 'oracle'}`` (see
    ``config.get_model_eval_layout``). ``questionnaire_dirs`` maps display name ->
    folder basename (defaults to ``config.EVAL_QUESTIONNAIRE_DIRS``). Per model,
    scores are read from
    ``<root>/metric=<subdir>/oracle=<oracle>/<model>/{patient_id}.csv``.
    Returns ``{display_name: df-or-None}``.
    """
    from .config import EVAL_QUESTIONNAIRE_DIRS, eval_csv_dir
    questionnaire_dirs = questionnaire_dirs or EVAL_QUESTIONNAIRE_DIRS
    out: dict = {}
    for label, sub in questionnaire_dirs.items():
        model_folders = {
            m: eval_csv_dir(model_layout[m]["root"], model_layout[m]["oracle"], sub, m)
            for m in model_list
            if m in model_layout
        }
        df = _load_eval_for_models(model_folders)
        if df is not None and len(df) > 0:
            df = add_model_metadata_columns(df)
            if model_order:
                df = apply_model_order(df, model_order=model_order)
        else:
            df = None
        out[label] = df
    return out


def build_test_cases(eval_results: dict, oracle_metric_map: dict = ORACLE_METRIC_MAP):
    """Build ``[(display_name, df, metric_col)]`` for every metric that has data."""
    cases = []
    for _, (display_name, metric_col) in oracle_metric_map.items():
        df = eval_results.get(display_name)
        if df is not None and metric_col in df.columns:
            cases.append((display_name, df, metric_col))
    return cases


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         MODEL SELECTION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _select_best_models_per_experiment(
    df: pd.DataFrame,
    metric: str,
    metric_name: Optional[str] = None,
) -> pd.DataFrame:
    """Best model per ExperimentGroup: highest mean, then lowest std, then earliest iter."""
    empty_cols = [
        "Metric", "ExperimentGroup", "BestModel", "Mean", "Std", "N",
        "LookAhead", "OracleGroup", "Iteration",
    ]
    if df is None or len(df) == 0 or metric not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    meta = add_model_metadata_columns(df.copy())
    grouped = (
        meta.groupby(["ExperimentGroup", "Model"], dropna=False, observed=True)[metric]
        .agg(["count", "mean", "std"]).reset_index()
    )
    grouped = grouped.merge(
        meta[["Model", "LookAhead", "OracleGroup", "Iteration"]].drop_duplicates("Model"),
        on="Model", how="left",
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["_iter"] = grouped["Iteration"].fillna(10**9)
    grouped = grouped.sort_values(
        by=["ExperimentGroup", "mean", "std", "_iter", "Model"],
        ascending=[True, False, True, True, True],
    )
    best = grouped.groupby("ExperimentGroup", as_index=False, observed=True).first()
    best = best.rename(columns={"Model": "BestModel", "count": "N", "mean": "Mean", "std": "Std"})
    best["Metric"] = metric_name or metric
    return best[empty_cols]


def select_best_models_by_own_oracle(
    test_cases: List[Tuple[str, pd.DataFrame, str]],
    oracle_metric_map: dict = ORACLE_METRIC_MAP,
    baseline: str = "Base",
):
    """Pick the best iteration per ExperimentGroup using each group's own oracle.

    Each DPO experiment group (e.g. ``L0_WAI``) is judged by its training oracle
    (``WAI-SR``). ``GRPO_Exp3`` uses ``Q1+Q2``. Base is always included.

    Returns ``(selected_models: List[str], summary_df: DataFrame)``.
    """
    lookup = {name: (df, col) for name, df, col in test_cases}
    discovered = set()
    for _, df, _ in test_cases:
        if df is not None:
            discovered.update(
                g for g in df["ExperimentGroup"].dropna().unique() if str(g) != baseline
            )

    summaries, selected = [], [baseline]
    for group in sorted(discovered):
        oracle_key = "Q1Q2" if group in ("GRPO_Exp3", "PTO_Exp3") else str(group).split("_", 1)[-1]
        if oracle_key not in oracle_metric_map:
            continue
        display_name, metric_col = oracle_metric_map[oracle_key]
        if display_name not in lookup:
            continue
        oracle_df, _ = lookup[display_name]
        if oracle_df is None:
            continue
        sub = oracle_df[oracle_df["ExperimentGroup"] == group]
        if len(sub) == 0:
            continue
        best = _select_best_models_per_experiment(sub, metric_col, metric_name=display_name)
        if len(best) > 0:
            summaries.append(best)
            selected.append(best.iloc[0]["BestModel"])

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return selected, summary


def filter_to_models(data, models, baseline: str = "Base"):
    """Filter a DataFrame OR a test_cases list to only the specified models (+ baseline)."""
    keep = {str(m) for m in models} | {str(baseline)}
    if isinstance(data, pd.DataFrame):
        return data[data["Model"].astype(str).isin(keep)].copy()
    if isinstance(data, list):
        return [
            (name, df[df["Model"].astype(str).isin(keep)].copy() if df is not None else None, col)
            for name, df, col in data
        ]
    raise TypeError(f"filter_to_models expects DataFrame or list, got {type(data)}")


def build_merged_metrics(test_cases):
    """Inner-join every metric on ``(Model, patient_id)`` for correlation analysis.

    Returns a wide DataFrame with one column per metric, or ``None`` when fewer
    than 2 metrics have data.
    """
    valid = [
        (name, df, col)
        for name, df, col in test_cases
        if df is not None and len(df) > 0 and col in df.columns
    ]
    if len(valid) < 2:
        return None

    _, first_df, first_col = valid[0]
    merged = first_df[["Model", "patient_id", first_col]].copy()
    for _, df, col in valid[1:]:
        merged = pd.merge(
            merged,
            df[["Model", "patient_id", col]].copy(),
            on=["Model", "patient_id"], how="inner",
        )
    return merged[[col for _, _, col in valid]]
