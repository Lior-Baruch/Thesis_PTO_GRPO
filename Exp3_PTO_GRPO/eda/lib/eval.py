"""
eval.py — Async oracle pipeline used by ``Run_Eval.ipynb``.

The questionnaires module is imported lazily so that the rest of the EDA still
imports cleanly even when an environment doesn't have it on ``sys.path``.
"""

import asyncio
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONCURRENCY, EVAL_MODEL, EVAL_TEMPERATURE, MAX_RETRIES, eval_csv_dir
from .data import reconstruct_conversation_text


EVAL_CODE_AVAILABLE = False
try:
    from questionnaires import (
        QuestionnaireID, get_prompt_eval_questionnaire, parse_json_response,
        Q1_LABELS, Q2_LABELS, WAI_SR_LABELS, WAI_SR_SUBSCALES,
        CSQ8_LABELS, MI_SAT_LABELS,
        MITI_GLOBAL_LABELS, MITI_BEHAVIOR_LABELS,
        PCT_GLOBAL_LABELS, PCT_BEHAVIOR_LABELS,
        MICI_GLOBAL_LABELS, MICI_BEHAVIOR_LABELS,
        _count_therapist_utterances,
    )
    EVAL_CODE_AVAILABLE = True
except ImportError:
    QuestionnaireID = None  # type: ignore[assignment]


def _require_eval_code() -> None:
    if not EVAL_CODE_AVAILABLE:
        raise RuntimeError(
            "questionnaires module not available — eval pipeline disabled. "
            "Ensure Exp3_PTO_GRPO/code/questionnaires.py is reachable on sys.path "
            "(eda/lib/__init__.py prepends the experiment's code/ dir automatically)."
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          ASYNC OPENAI CALL                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


async def call_openai_json(
    client,
    prompt: str,
    schema: dict,
    schema_name: str = "evaluation",
    model: str = EVAL_MODEL,
    temperature: float = EVAL_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """OpenAI chat completion with structured JSON response and exponential back-off."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, seed=42, max_tokens=512,
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            ROW BUILDERS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


if EVAL_CODE_AVAILABLE:

    # Simple questionnaires (Q1, Q2, CSQ8, MI_SAT): row = scores keyed by label,
    # plus mean + total aggregates. Driven by a per-questionnaire spec so the
    # six-near-identical _build_*_row functions collapse to one factory.
    _SIMPLE_ROW_SPECS = {
        QuestionnaireID.Q1:     (Q1_LABELS,     "Q1_Mean",   "Q1_Total"),
        QuestionnaireID.Q2:     (Q2_LABELS,     "Q2_Mean",   "Q2_Total"),
        QuestionnaireID.CSQ8:   (CSQ8_LABELS,   "CSQ8_Mean", "CSQ8_Total"),
        QuestionnaireID.MI_SAT: (MI_SAT_LABELS, "MI_Mean",   "MI_Total"),
    }

    def _build_simple_row(scores: dict, labels: list, mean_col: str, total_col: str) -> dict:
        row = {k: scores.get(k, np.nan) for k in labels}
        vals = [row[k] for k in labels]
        row[mean_col] = float(np.nanmean(vals))
        row[total_col] = float(np.nansum(vals))
        return row

    def _build_wai_sr_row(scores: dict) -> dict:
        row = {k: scores.get(k, np.nan) for k in WAI_SR_LABELS}
        for sub, items in WAI_SR_SUBSCALES.items():
            row[f"{sub}_Mean"] = float(np.nanmean([row.get(i, np.nan) for i in items]))
        row["WAI_TotalMean"] = float(np.nanmean([row[k] for k in WAI_SR_LABELS]))
        row["WAI_TotalSum"] = float(np.nansum([row[k] for k in WAI_SR_LABELS]))
        return row

    def _build_miti_row(scores: dict) -> pd.DataFrame:
        # MITI is asymmetric: globals get a mean, behaviors get a total.
        row: dict = {}
        for k in MITI_GLOBAL_LABELS:
            row[k] = scores.get(k, np.nan)
        row["MITI_GlobalMean"] = float(np.nanmean([row[k] for k in MITI_GLOBAL_LABELS]))
        for k in MITI_BEHAVIOR_LABELS:
            row[k] = scores.get(k, np.nan)
        bvals = [row[k] for k in MITI_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
        row["MITI_BehaviorTotal"] = float(np.nansum(bvals)) if bvals else np.nan
        return pd.DataFrame([row])

    def _build_pct_row(scores: dict) -> dict:
        # PCT (Patient Change Talk): 3 globals get a mean; 3 patient-utterance
        # counts get a total + the orthogonal ChangeProp = CT / (CT + ST).
        row = {k: scores.get(k, np.nan) for k in PCT_GLOBAL_LABELS}
        row["PCT_GlobalMean"] = float(np.nanmean([row[k] for k in PCT_GLOBAL_LABELS]))
        for k in PCT_BEHAVIOR_LABELS:
            row[k] = scores.get(k, np.nan)
        bvals = [row[k] for k in PCT_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
        row["PCT_BehaviorTotal"] = float(np.nansum(bvals)) if bvals else np.nan
        ct, st = row.get("PCT_ChangeTalk", np.nan), row.get("PCT_SustainTalk", np.nan)
        denom = (0 if pd.isna(ct) else ct) + (0 if pd.isna(st) else st)
        row["PCT_ChangeProp"] = float(ct / denom) if denom > 0 else np.nan
        return row

    def _build_mici_row(scores: dict, n_th_turns: int) -> dict:
        # MICI (MI-Inconsistent): severity global + harmful-behavior counts.
        # Higher = worse. Rate = total inconsistent behaviors per therapist turn.
        row = {k: scores.get(k, np.nan) for k in MICI_GLOBAL_LABELS}
        for k in MICI_BEHAVIOR_LABELS:
            row[k] = scores.get(k, np.nan)
        bvals = [row[k] for k in MICI_BEHAVIOR_LABELS if not pd.isna(row.get(k, np.nan))]
        total = float(np.nansum(bvals)) if bvals else np.nan
        row["MICI_BehaviorTotal"] = total
        row["MICI_Rate"] = float(total / n_th_turns) if n_th_turns else np.nan
        op = row.get("MICI_OverPraise", np.nan)
        row["MICI_OverPraiseRate"] = float(op / n_th_turns) if (n_th_turns and not pd.isna(op)) else np.nan
        return row

    def _build_row(qid_enum, scores: dict, conv_str: str = "") -> pd.DataFrame:
        """Dispatch to the right row builder based on questionnaire id.

        Returns a single-row DataFrame for any questionnaire (MITI's builder
        already returns a DataFrame; simple/WAI return dicts wrapped here).
        ``conv_str`` is used only by MICI to compute per-therapist-turn rates.
        """
        if qid_enum in _SIMPLE_ROW_SPECS:
            labels, mean_col, total_col = _SIMPLE_ROW_SPECS[qid_enum]
            return pd.DataFrame([_build_simple_row(scores, labels, mean_col, total_col)])
        if qid_enum == QuestionnaireID.WAI_SR:
            return pd.DataFrame([_build_wai_sr_row(scores)])
        if qid_enum == QuestionnaireID.MITI:
            return _build_miti_row(scores)
        if qid_enum == QuestionnaireID.PCT:
            return pd.DataFrame([_build_pct_row(scores)])
        if qid_enum == QuestionnaireID.MICI:
            return pd.DataFrame([_build_mici_row(scores, _count_therapist_utterances(conv_str))])
        # Unknown questionnaire — return raw scores dict as a single row.
        return pd.DataFrame([scores])

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║                       PER-CONVERSATION EVAL                            ║
    # ╚══════════════════════════════════════════════════════════════════════════╝

    async def evaluate_conversation(
        client,
        conversation,
        questionnaire_id,
        model: str = EVAL_MODEL,
        eval_temperature: float = EVAL_TEMPERATURE,
    ) -> Optional[pd.DataFrame]:
        """Score one conversation with one questionnaire; return a 1-row DataFrame."""
        conv_str = reconstruct_conversation_text(conversation)
        qid_enum = (
            questionnaire_id
            if isinstance(questionnaire_id, QuestionnaireID)
            else QuestionnaireID(questionnaire_id)
        )
        try:
            ed = get_prompt_eval_questionnaire(questionnaire=questionnaire_id, conversation=conv_str)
            resp = await call_openai_json(
                client,
                prompt=ed["prompt"], schema=ed["schema"],
                schema_name=f"questionnaire_{qid_enum.value}_evaluation",
                model=model, temperature=eval_temperature,
            )
            result = parse_json_response(
                response_content=resp, questionnaire_id=questionnaire_id, labels=ed["labels"]
            )
            return _build_row(qid_enum, result["scores_dict"], conv_str)
        except Exception as e:
            print(f"Error evaluating with questionnaire {qid_enum.value}: {e}")
            return None

    def build_default_eval_configs(config) -> list:
        """Build the default eval-config list from an :class:`EDAConfig`.

        Each config carries the questionnaire's folder *basename* (``q_subdir``),
        not an absolute folder — the writer joins it under each model's own
        method root (see :func:`run_all_evaluations_async`).
        """
        from .config import EVAL_QUESTIONNAIRE_DIRS
        specs = [
            ("CSQ-8", QuestionnaireID.CSQ8),
            ("WAI-SR", QuestionnaireID.WAI_SR),
            ("MI-SAT", QuestionnaireID.MI_SAT),
            ("MITI", QuestionnaireID.MITI),
            ("Q1", QuestionnaireID.Q1),
            ("Q2", QuestionnaireID.Q2),
            ("PCT", QuestionnaireID.PCT),
            ("MICI", QuestionnaireID.MICI),
        ]
        return [
            {
                "name": n, "id": q,
                "q_subdir": EVAL_QUESTIONNAIRE_DIRS[n],
                "model": config.eval_model,
                "eval_temperature": config.eval_temp,
            }
            for n, q in specs
        ]

    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║                            BATCH RUNNERS                               ║
    # ╚══════════════════════════════════════════════════════════════════════════╝

    async def _process_one_questionnaire(
        client, combined_data: pd.DataFrame, qconfig: dict, model_layout: dict,
    ) -> dict:
        """Score every conversation with one questionnaire; skip already-written CSVs.

        Each model is written under its own labelled folder:
        ``<root>/metric=<q_subdir>/oracle=<oracle>/<model>/<patient_id>.csv`` —
        root + oracle come from ``model_layout[model]``. Models with no entry in
        ``model_layout`` are skipped (with a warning).
        """
        q_id = qconfig["id"]
        q_subdir = qconfig["q_subdir"]
        name = qconfig.get("name", f"Q{q_id.value if isinstance(q_id, QuestionnaireID) else q_id}")
        model_id = qconfig.get("model", EVAL_MODEL)
        eval_temp = qconfig.get("eval_temperature", EVAL_TEMPERATURE)
        concurrency = qconfig.get("concurrency", DEFAULT_CONCURRENCY)
        verbose = qconfig.get("verbose", True)

        stats = {"completed": 0, "skipped_existing": 0, "skipped_incomplete": 0, "errors": 0}
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def _process(row, mf):
            out_fp = os.path.join(mf, f"{row['id']}.csv")
            if os.path.exists(out_fp):
                async with lock:
                    stats["skipped_existing"] += 1
                return
            try:
                rdf = await evaluate_conversation(
                    client, conversation=row["conversation"], questionnaire_id=q_id,
                    model=model_id, eval_temperature=eval_temp,
                )
                if rdf is None or rdf.isnull().values.any():
                    async with lock:
                        stats["skipped_incomplete"] += 1
                    return
                rdf.to_csv(out_fp, index=False)
                async with lock:
                    stats["completed"] += 1
            except Exception as e:
                print(f"Error: {name} {row['Model']}/{row['id']}: {e}")
                async with lock:
                    stats["errors"] += 1

        async def _bounded(row, mf):
            async with sem:
                await _process(row, mf)

        tasks = []
        for model in combined_data["Model"].unique():
            entry = model_layout.get(str(model))
            if entry is None:
                if verbose:
                    print(f"  (skip {model}: no entry in model_layout)")
                continue
            mf = eval_csv_dir(entry["root"], entry["oracle"], q_subdir, str(model))
            os.makedirs(mf, exist_ok=True)
            if verbose:
                print(f"Evaluating {name} for model: {model}")
            for _, row in combined_data[combined_data["Model"] == model].iterrows():
                tasks.append(asyncio.create_task(_bounded(row, mf)))
        if tasks:
            await asyncio.gather(*tasks)

        total = sum(stats.values())
        print(
            f"{name}: {stats['completed']}/{total} new, "
            f"{stats['skipped_existing']} existing, {stats['errors']} errors"
        )
        return stats

    async def run_all_evaluations_async(
        client, combined_data: pd.DataFrame, configs: list, model_layout: dict,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> dict:
        """Run ``_process_one_questionnaire`` for every config sequentially.

        ``model_layout`` maps each model name to ``{'root', 'oracle'}`` (use
        :func:`config.get_model_eval_layout`), so scores land under
        ``<root>/metric=<M>/oracle=<O>/<model>/``.
        """
        results = {}
        for c in configs:
            c = {**c, "concurrency": concurrency}
            results[c["name"]] = await _process_one_questionnaire(
                client, combined_data, c, model_layout,
            )
        return results
