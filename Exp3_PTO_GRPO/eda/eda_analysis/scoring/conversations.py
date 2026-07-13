"""
conversations.py — loading conversations for scoring + model-name metadata parsing.

Two concerns:
1. **Conversations** — read per-model CSVs into a single DataFrame so the oracle
   pipeline can score them (``load_data``/``combine_data``/``reconstruct_conversation_text``).
2. **Model metadata** — parse a registry model name into
   ``{LookAhead, OracleGroup, Iteration, ExperimentGroup}`` columns
   (``parse_model_metadata``/``add_model_metadata_columns``).

Note this is the *scoring-side* loader (keyed on the registry's model names); the
analysis-side backbone with persona recovery is :mod:`eda_analysis.data`.
"""

import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .registry import ORACLE_TOKEN_ALIASES


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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       MODEL METADATA PARSING                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


_MODEL_NAME_PATTERN  = re.compile(r"^L(?P<lookahead>\d+)_(?P<oracle>[A-Za-z0-9_]+)_V(?P<iteration>\d+)$")
_GRPOExp3_NAME_PATTERN = re.compile(r"^GRPOExp3_LA(?P<la>\d+)_(?:(?P<base>Base)|I(?P<iter>\d+))$")
_PTOExp3_NAME_PATTERN  = re.compile(r"^PTOExp3_LA(?P<la>\d+)_(?:(?P<base>Base)|I(?P<iter>\d+))$")


def _normalize_oracle_token(token: str, *, strict: bool = False) -> str:
    """Canonicalize oracle name tokens to the canonical oracle keys (WAI/CSQ8/Q1Q2/MI_SAT/MITI).

    Aliases live in ``registry.ORACLE_TOKEN_ALIASES``. Unknown tokens fall through
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
            f"Add it to registry.ORACLE_TOKEN_ALIASES."
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
        return {"LookAhead": int(m.group("la")), "OracleGroup": oracle, "Iteration": iteration, "ExperimentGroup": "GRPO_Exp3"}

    m = _PTOExp3_NAME_PATTERN.match(model_name)
    if m:
        is_base = bool(m.group("base"))
        iteration = 0 if is_base else int(m.group("iter"))
        oracle = "Base" if is_base else "Q1Q2"
        return {"LookAhead": int(m.group("la")), "OracleGroup": oracle, "Iteration": iteration, "ExperimentGroup": "PTO_Exp3"}

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
