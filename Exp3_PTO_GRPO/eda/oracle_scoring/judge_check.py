"""
judge_check.py — measurement-validity re-scoring: oracle repeatability (ICC) + second-judge
agreement. Powers ``Judge_Reliability.ipynb``.

Buys down LIMITATIONS.md §1 (judge reliability not measured) and §2 (patient = oracle coupling)
on a SUBSET, cheaply:

1. **Repeatability / ICC** — re-score the same conversations N times with the PRIMARY oracle
   (gpt-4o-mini, same temperature, *different seeds* — the pipeline's fixed ``seed=42`` would
   make reps identical by design, so reps get ``seed=1000+rep``). → per-metric ICC(2,1) +
   mean |Δ| between reps.
2. **Second judge** — score the same conversations once with a DIFFERENT judge (pluggable
   provider: ``openai`` or ``anthropic``/Claude). → per-metric correlation + bias vs the primary
   oracle, and (the defense-critical check) whether the PTO-vs-GRPO endpoint contrast is
   preserved under the second judge.

Outputs are kept OUT of the real ``eval_scores/`` tree:
    ``data/judge_check/<judge_tag>/rep=<r>/metric=<M>/oracle=<O>/<Model>/<id>.csv``
(same per-conversation CSV shape as Run_Eval, so loaders are shared). Resume-safe: existing
CSVs are skipped, like Run_Eval.

Anthropic judge notes (per the Claude API docs, 2026-06):
- Structured output via ``output_config.format`` (json_schema) — the response's first text block
  is guaranteed valid JSON. No assistant prefill, no tool-choice gymnastics.
- Numeric bounds (minimum/maximum) and array length constraints (minItems/maxItems) are NOT
  supported in Claude json_schema — stripped before sending; ``parse_json_response`` still
  length-validates client-side.
- ``temperature`` is rejected on the newest models (Opus 4.7+/Sonnet 5) — omitted for Claude.
- Requires the ``anthropic`` package + a key in env ``ANTHROPIC_API_KEY`` or
  ``anthropic_key.txt`` at the experiment root (beside openai_key.txt).
"""

import asyncio
import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import WORKSPACE_ROOT
from .config import EVAL_MODEL, EVAL_TEMPERATURE, MAX_RETRIES, EVAL_QUESTIONNAIRE_DIRS
from .data import reconstruct_conversation_text

# Reuse the questionnaire prompt/parse/row machinery from the primary pipeline.
from . import eval as _eval

JUDGE_CHECK_ROOT = os.path.join(WORKSPACE_ROOT, "data", "judge_check")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              JUDGE SPEC                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass
class JudgeSpec:
    """One judge configuration (provider + model)."""
    provider: str                      # "openai" | "anthropic"
    model: str                         # e.g. "gpt-4o-mini-2024-07-18" | "claude-haiku-4-5"
    temperature: Optional[float] = None  # None = omit (required for newest Claude models)
    max_tokens: int = 1024

    @property
    def tag(self) -> str:
        """Filesystem-safe judge folder name, e.g. ``anthropic_claude-haiku-4-5``."""
        return re.sub(r"[^A-Za-z0-9.\-]+", "_", f"{self.provider}_{self.model}")


PRIMARY_JUDGE = JudgeSpec(provider="openai", model=EVAL_MODEL, temperature=EVAL_TEMPERATURE)


def _read_key_file(fname: str) -> Optional[str]:
    fp = os.path.join(WORKSPACE_ROOT, fname)
    if os.path.exists(fp):
        with open(fp, encoding="utf-8") as f:
            return f.read().strip()
    return None


def init_judge_client(judge: JudgeSpec):
    """Async client for a judge. OpenAI: env ``OPENAI_API_KEY`` → ``openai_key.txt``.
    Anthropic: env ``ANTHROPIC_API_KEY`` → ``anthropic_key.txt`` (experiment root)."""
    if judge.provider == "openai":
        from openai import AsyncOpenAI
        key = os.environ.get("OPENAI_API_KEY") or _read_key_file("openai_key.txt")
        if not key:
            raise RuntimeError("No OpenAI key (env OPENAI_API_KEY or openai_key.txt).")
        return AsyncOpenAI(api_key=key)
    if judge.provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise RuntimeError("`pip install anthropic` to use a Claude judge.") from e
        key = os.environ.get("ANTHROPIC_API_KEY") or _read_key_file("anthropic_key.txt")
        if not key:
            raise RuntimeError("No Anthropic key (env ANTHROPIC_API_KEY or anthropic_key.txt "
                               "at the experiment root, beside openai_key.txt).")
        return AsyncAnthropic(api_key=key)
    raise ValueError(f"Unknown judge provider {judge.provider!r}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        PROVIDER-DISPATCHED JSON CALL                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def call_openai_json_seeded(client, prompt: str, schema: dict, *, schema_name: str,
                                  model: str, temperature: float, seed: int,
                                  max_retries: int = MAX_RETRIES) -> dict:
    """``eval.call_openai_json`` with a CONTROLLABLE seed (the primary pipeline pins seed=42,
    which would make repeatability reps identical by construction)."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, seed=seed, max_tokens=512,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response")
            return json.loads(content)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise ValueError(f"Failed after {max_retries} attempts: {e}")
            await asyncio.sleep(2 ** attempt)


def _strip_unsupported_constraints(schema: dict) -> dict:
    """Claude json_schema structured outputs reject numeric bounds + array length constraints —
    strip ``minimum``/``maximum``/``minItems``/``maxItems`` recursively (validation of counts is
    re-done client-side by ``parse_json_response``)."""
    s = copy.deepcopy(schema)

    def walk(node):
        if isinstance(node, dict):
            for k in ("minimum", "maximum", "minItems", "maxItems", "multipleOf"):
                node.pop(k, None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(s)
    return s


async def call_anthropic_json(client, prompt: str, schema: dict, *, model: str,
                              max_tokens: int = 1024,
                              max_retries: int = MAX_RETRIES) -> dict:
    """Claude structured output: ``output_config.format`` json_schema → first text block is
    guaranteed valid JSON. No ``temperature`` (rejected on Opus 4.7+/Sonnet 5); determinism is
    not required — cross-judge agreement is measured over 96-conv means."""
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                output_config={"format": {"type": "json_schema",
                                          "schema": _strip_unsupported_constraints(schema)}},
                messages=[{"role": "user", "content": prompt}],
            )
            if response.stop_reason == "refusal":
                raise ValueError("Claude refused the request (stop_reason=refusal)")
            text = next((b.text for b in response.content if b.type == "text"), "")
            if not text.strip():
                raise ValueError("Empty response")
            return json.loads(text)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise ValueError(f"Failed after {max_retries} attempts: {e}")
            await asyncio.sleep(2 ** attempt)


async def evaluate_conversation_with_judge(client, judge: JudgeSpec, conversation,
                                           questionnaire_id, *, seed: int = 42
                                           ) -> Optional[pd.DataFrame]:
    """Score one conversation with one questionnaire under an arbitrary judge.
    Mirrors ``eval.evaluate_conversation`` but dispatches on provider."""
    if not _eval.EVAL_CODE_AVAILABLE:
        raise RuntimeError("questionnaires module not importable — run from eda/ with code/ on sys.path")
    conv_str = reconstruct_conversation_text(conversation)
    qid_enum = (questionnaire_id if isinstance(questionnaire_id, _eval.QuestionnaireID)
                else _eval.QuestionnaireID(questionnaire_id))
    try:
        ed = _eval.get_prompt_eval_questionnaire(questionnaire=questionnaire_id, conversation=conv_str)
        if judge.provider == "openai":
            resp = await call_openai_json_seeded(
                client, ed["prompt"], ed["schema"],
                schema_name=f"questionnaire_{qid_enum.value}_evaluation",
                model=judge.model,
                temperature=judge.temperature if judge.temperature is not None else EVAL_TEMPERATURE,
                seed=seed)
        else:
            resp = await call_anthropic_json(client, ed["prompt"], ed["schema"],
                                             model=judge.model, max_tokens=judge.max_tokens)
        result = _eval.parse_json_response(response_content=resp,
                                           questionnaire_id=questionnaire_id, labels=ed["labels"])
        return _eval._build_row(qid_enum, result["scores_dict"], conv_str)
    except Exception as e:
        print(f"  [judge_check] error ({judge.tag}, Q{qid_enum.value}): {e}")
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          BATCH RE-SCORING RUNNER                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def judge_out_dir(judge_tag: str, rep: int, metric_subdir: str, oracle: str, model: str) -> str:
    return os.path.join(JUDGE_CHECK_ROOT, judge_tag, f"rep={rep}",
                        f"metric={metric_subdir}", f"oracle={oracle}", model)


async def run_judge_scoring(judge: JudgeSpec, combined_data: pd.DataFrame,
                            questionnaire_names: List[str], model_layout: Dict[str, Dict[str, str]],
                            *, rep: int = 0, concurrency: int = 16,
                            subset_n: Optional[int] = None) -> dict:
    """Score every (model, conversation, questionnaire) in ``combined_data`` under ``judge``.

    - ``combined_data``: Run_Eval-style frame (``Model``, ``id``, ``conversation`` columns).
    - ``questionnaire_names``: display names, e.g. ``["Q1", "Q2", "MICI"]``.
    - ``model_layout``: ``config.get_model_eval_layout()`` — used only for the oracle label
      (output root is ``data/judge_check/``, never the real eval_scores).
    - ``rep``: repetition index → its own folder + (openai) its own seed ``1000+rep``.
    - ``subset_n``: score only the first N conversations per model (cost lever).
    Resume-safe (skips existing CSVs). Returns a stats dict.
    """
    from questionnaires import QuestionnaireID  # canonical enum (code/ on sys.path)
    name_to_qid = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
                   "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
                   "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
                   "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}

    client = init_judge_client(judge)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"completed": 0, "skipped_existing": 0, "errors": 0}
    seed = 1000 + rep   # openai only; Claude path ignores it

    async def _one(row, qname, out_dir):
        out_fp = os.path.join(out_dir, f"{row['id']}.csv")
        if os.path.exists(out_fp):
            async with lock:
                stats["skipped_existing"] += 1
            return
        async with sem:
            rdf = await evaluate_conversation_with_judge(
                client, judge, row["conversation"], name_to_qid[qname], seed=seed)
        if rdf is None or rdf.isnull().values.any():
            async with lock:
                stats["errors"] += 1
            return
        rdf.to_csv(out_fp, index=False)
        async with lock:
            stats["completed"] += 1

    tasks = []
    for model in combined_data["Model"].unique():
        entry = model_layout.get(str(model))
        oracle = entry["oracle"] if entry else "none"
        sub = combined_data[combined_data["Model"] == model]
        if subset_n is not None:
            sub = sub.iloc[:subset_n]
        for qname in questionnaire_names:
            out_dir = judge_out_dir(judge.tag, rep, EVAL_QUESTIONNAIRE_DIRS[qname], oracle, str(model))
            os.makedirs(out_dir, exist_ok=True)
            for _, row in sub.iterrows():
                tasks.append(asyncio.create_task(_one(row, qname, out_dir)))
    if tasks:
        await asyncio.gather(*tasks)
    print(f"[judge_check] {judge.tag} rep={rep}: {stats['completed']} new, "
          f"{stats['skipped_existing']} existing, {stats['errors']} errors")
    return stats


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          LOADING + STATISTICS                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# metric display name -> (metric_subdir, per-conv value column) — the columns the EDA uses.
JUDGE_METRIC_COLS = {
    "Q1": ("Q1", "Q1_Mean"), "Q2": ("Q2", "Q2_Mean"),
    "WAI-SR": ("WAI_SR", "WAI_TotalMean"), "CSQ-8": ("CSQ8", "CSQ8_Mean"),
    "MI-SAT": ("MI_SAT", "MI_Mean"), "MITI": ("MITI", "MITI_GlobalMean"),
    "PCT": ("PCT", "PCT_ChangeProp"), "MICI": ("MICI", "MICI_Rate"),
}


def load_judge_scores(judge_tag: str, *, reps: Optional[List[int]] = None) -> pd.DataFrame:
    """Tidy long frame of everything a judge has scored: one row per
    (rep, metric, model, conversation) -> value. Discovers reps/metrics/models from disk."""
    root = os.path.join(JUDGE_CHECK_ROOT, judge_tag)
    rows = []
    if not os.path.isdir(root):
        return pd.DataFrame(columns=["judge", "rep", "metric", "oracle", "model", "file_index", "value"])
    subdir_to_name = {v[0]: k for k, v in JUDGE_METRIC_COLS.items()}
    for rep_dir in sorted(os.listdir(root)):
        m = re.match(r"rep=(\d+)$", rep_dir)
        if not m:
            continue
        rep = int(m.group(1))
        if reps is not None and rep not in reps:
            continue
        for mdir in os.listdir(os.path.join(root, rep_dir)):
            mm = re.match(r"metric=(.+)$", mdir)
            if not mm or mm.group(1) not in subdir_to_name:
                continue
            name = subdir_to_name[mm.group(1)]
            val_col = JUDGE_METRIC_COLS[name][1]
            for odir in os.listdir(os.path.join(root, rep_dir, mdir)):
                om = re.match(r"oracle=(.+)$", odir)
                if not om:
                    continue
                for model in os.listdir(os.path.join(root, rep_dir, mdir, odir)):
                    ddir = os.path.join(root, rep_dir, mdir, odir, model)
                    if not os.path.isdir(ddir):
                        continue
                    for fn in os.listdir(ddir):
                        stem, ext = os.path.splitext(fn)
                        if ext != ".csv" or not stem.isdigit():
                            continue
                        try:
                            df = pd.read_csv(os.path.join(ddir, fn))
                        except Exception:
                            continue
                        if len(df) and val_col in df.columns:
                            rows.append({"judge": judge_tag, "rep": rep, "metric": name,
                                         "oracle": om.group(1), "model": model,
                                         "file_index": int(stem),
                                         "value": float(df[val_col].iloc[0])})
    return pd.DataFrame(rows)


def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1) — two-way random effects, absolute agreement, single rater (Shrout & Fleiss).
    ``matrix``: n_targets × k_raters (here: conversations × reps). NaN rows dropped."""
    m = np.asarray(matrix, float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 3 or k < 2:
        return np.nan
    mean_t = m.mean(axis=1)
    mean_r = m.mean(axis=0)
    grand = m.mean()
    ssr = k * ((mean_t - grand) ** 2).sum()                  # targets (rows)
    ssc = n * ((mean_r - grand) ** 2).sum()                  # raters (columns)
    sse = ((m - mean_t[:, None] - mean_r[None, :] + grand) ** 2).sum()
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom else np.nan


def repeatability_table(judge_long: pd.DataFrame) -> pd.DataFrame:
    """Per (metric, model): ICC(2,1) across reps + mean |Δ| between rep pairs + n convs."""
    rows = []
    for (metric, model), g in judge_long.groupby(["metric", "model"]):
        piv = g.pivot_table(index="file_index", columns="rep", values="value")
        if piv.shape[1] < 2:
            continue
        m = piv.to_numpy(float)
        pair_absdiff = []
        cols = list(range(piv.shape[1]))
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                d = np.abs(m[:, i] - m[:, j])
                pair_absdiff.append(np.nanmean(d))
        rows.append({"metric": metric, "model": model, "n_convs": int(piv.shape[0]),
                     "n_reps": int(piv.shape[1]), "icc_2_1": round(icc_2_1(m), 3),
                     "mean_abs_diff": round(float(np.nanmean(pair_absdiff)), 3)})
    return pd.DataFrame(rows).sort_values(["metric", "model"]).reset_index(drop=True)


def agreement_table(judge_long: pd.DataFrame, primary_long: pd.DataFrame) -> pd.DataFrame:
    """Per (metric, model): second judge vs primary — Pearson r, Spearman ρ, mean bias
    (judge − primary), n. Join on (metric, model, file_index)."""
    from scipy import stats as sps
    j = judge_long.groupby(["metric", "model", "file_index"])["value"].mean().rename("judge")
    p = primary_long.groupby(["metric", "model", "file_index"])["value"].mean().rename("primary")
    merged = pd.concat([j, p], axis=1).dropna().reset_index()
    rows = []
    for (metric, model), g in merged.groupby(["metric", "model"]):
        if len(g) < 3:
            continue
        pear = float(np.corrcoef(g["judge"], g["primary"])[0, 1]) if g["judge"].std() and g["primary"].std() else np.nan
        rho = float(sps.spearmanr(g["judge"], g["primary"]).statistic)
        rows.append({"metric": metric, "model": model, "n": len(g),
                     "pearson_r": round(pear, 3), "spearman_rho": round(rho, 3),
                     "bias_judge_minus_primary": round(float((g["judge"] - g["primary"]).mean()), 3)})
    return pd.DataFrame(rows).sort_values(["metric", "model"]).reset_index(drop=True)


def contrast_preservation(judge_long: pd.DataFrame, primary_long: pd.DataFrame,
                          model_a: str, model_b: str, metric: str) -> dict:
    """THE defense check: does the second judge reproduce the primary oracle's A−B contrast?
    Paired by file_index (valid within the same model_iter: same persona shuffle)."""
    out = {"metric": metric, "model_a": model_a, "model_b": model_b}
    for name, src in (("judge", judge_long), ("primary", primary_long)):
        g = src[src.metric == metric]
        a = g[g.model == model_a].set_index("file_index")["value"]
        b = g[g.model == model_b].set_index("file_index")["value"]
        common = a.index.intersection(b.index)
        d = (a.loc[common] - b.loc[common]).astype(float)
        out[f"{name}_delta"] = round(float(d.mean()), 3) if len(d) else np.nan
        out[f"{name}_n"] = int(len(d))
    out["same_sign"] = (np.sign(out.get("judge_delta", np.nan)) ==
                        np.sign(out.get("primary_delta", np.nan)))
    return out
