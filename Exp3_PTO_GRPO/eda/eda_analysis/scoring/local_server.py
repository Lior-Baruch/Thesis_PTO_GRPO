"""local_server.py — run an open-weights grader behind an OpenAI-compatible endpoint.

The point of this module is that it adds no scoring code. vLLM (and llama.cpp's server,
and Ollama) speak the OpenAI protocol including ``response_format={"type":"json_schema"}``
via constrained decoding, so a locally-served Gemma is reachable through the ordinary
:class:`~eda_analysis.scoring.judge.JudgeSpec` path — same prompts, same parsing, same
validation, same resume-by-skipping-CSVs. All that is missing is (a) starting the process
and knowing when it is ready and (b) deciding whether the model is fit to grade at all.

Why (b) needs its own gate
--------------------------
``judge_plan.check_rubric_parity`` answers a *static* question — were the constraints we
stripped for Claude restated in prose. A local server strips nothing, so parity is trivially
clean and tells us nothing. The real risks with a small open-weights grader are empirical:

1. It ignores the schema and returns prose, or returns the wrong number of item scores.
2. Worse, it honours the schema perfectly and returns **degenerate** scores — every item a 4,
   near-zero variance across conversations. That parses, writes valid CSVs, and produces a
   judge that cannot tell any two arms apart. Nothing downstream would flag it; the
   agreement tables would just come back ~0 and look like a finding.

:func:`probe_rubrics` and :func:`probe_discrimination` catch both, for a handful of calls,
before committing to a 22k-cell sweep.

Typical use (Colab, where the GPU is otherwise idle during scoring)::

    with serve("google/gemma-3n-E4B-it", gpu_memory_utilization=0.85) as srv:
        judge = local_judge(srv.model, srv.base_url)
        probe_rubrics(judge, combined, METRICS)
        await run_judge_scoring(judge, combined, METRICS, layout, rep=0)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .judge import JudgeSpec, evaluate_conversation_with_judge, init_judge_client

__all__ = [
    "ServerHandle", "serve", "start_server", "local_judge", "wait_until_ready",
    "probe_rubrics", "probe_discrimination",
]

DEFAULT_PORT = 8000


@dataclass
class ServerHandle:
    """A running (or externally-managed) OpenAI-compatible server."""
    model: str
    base_url: str
    process: Optional[subprocess.Popen] = None
    log_path: Optional[str] = None

    def stop(self, timeout: float = 30.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def tail_log(self, n: int = 40) -> str:
        if not self.log_path or not os.path.exists(self.log_path):
            return "(no log)"
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])


def wait_until_ready(base_url: str, *, timeout: float = 900.0,
                     process: Optional[subprocess.Popen] = None,
                     poll_seconds: float = 3.0) -> None:
    """Block until ``GET {base_url}/models`` answers, or raise.

    Also watches *process*: a server that dies during weight loading (OOM is the usual
    reason) would otherwise leave the caller waiting out the whole timeout for a port that
    is never going to open.
    """
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/models"
    last_err = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode} before becoming ready — "
                f"check the log (OOM during weight load is the usual cause)")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = e
        time.sleep(poll_seconds)
    raise TimeoutError(f"server at {base_url} not ready after {timeout:.0f}s ({last_err})")


def start_server(model: str, *, port: int = DEFAULT_PORT,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 8192,
                 log_path: Optional[str] = None,
                 extra_args: Optional[Sequence[str]] = None, timeout: float = 900.0,
                 executable: str = "vllm") -> ServerHandle:
    """Launch a server and block until it answers; caller owns ``handle.stop()``.

    The notebook-friendly half of :func:`serve` — a ``with`` block cannot span notebook
    cells, and the whole point of the validation pass is to probe, look at the numbers, then
    decide whether to run the sweep. See :func:`serve` for the memory guidance.
    """
    base_url = f"http://localhost:{port}/v1"
    log_path = log_path or os.path.join(os.getcwd(), f"vllm_{port}.log")
    cmd = [executable, "serve", model, "--port", str(port),
           "--gpu-memory-utilization", str(gpu_memory_utilization),
           "--max-model-len", str(max_model_len), *(extra_args or [])]
    print("launching:", " ".join(cmd))
    print("log:", log_path)
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    handle = ServerHandle(model=model, base_url=base_url, process=proc, log_path=log_path)
    try:
        wait_until_ready(base_url, timeout=timeout, process=proc)
    except BaseException:
        print(handle.tail_log())
        handle.stop()
        raise
    print(f"ready: {model} @ {base_url}")
    return handle


@contextmanager
def serve(model: str, **kwargs):
    """Context-manager form of :func:`start_server` — stops the server on exit.

    ⚠ ``gpu_memory_utilization`` is a **pre-allocation**, not a ceiling that grows on demand:
    vLLM reserves that fraction of the card for weights + KV pool at startup. Two consequences:

    - Scoring on an otherwise-idle GPU (the S3 validation pass) wants it HIGH — 0.85 or so —
      because a bigger KV pool is what buys concurrency.
    - Sharing the card with a live trainer wants it LOW (~0.25) and the server started FIRST,
      because training memory is the spiky side (DPO's full-sequence logits over a 128k
      vocab) and it should get the slack.
    - On a 12 GB card, a bf16 4B model plus any meaningful pool does not fit. Use a quantized
      build or a bigger GPU; an over-budget request on the local Blackwell card hard-faults
      the machine rather than raising (see the root CLAUDE.md gotchas).

    Pass ``extra_args`` for backend-specific flags (quantization, dtype, guided-decoding
    backend). Set ``executable`` to point at a different OpenAI-compatible server binary.
    """
    handle = start_server(model, **kwargs)
    try:
        yield handle
    finally:
        handle.stop()
        print("server stopped")


def local_judge(model: str, base_url: str, *, max_tokens: int = 1024,
                temperature: Optional[float] = 0.0, tag: Optional[str] = None) -> JudgeSpec:
    """:class:`JudgeSpec` for a locally-served model.

    The default tag is ``local_<shorttag>`` (via ``roles.model_tag``) rather than the
    provider_model default, so the score-lake partition reads ``judge=local_gemma3nE4B``
    instead of ``judge=openai_compat_google_gemma-3n-E4B-it``. The tag is a directory name
    that ends up in every artifact path — keep it stable once you have scored anything.
    """
    from roles import model_tag
    return JudgeSpec(provider="openai_compat", model=model, base_url=base_url,
                     temperature=temperature, max_tokens=max_tokens,
                     tag_override=tag or f"local_{model_tag(model)}")


async def probe_rubrics(judge: JudgeSpec, combined: pd.DataFrame,
                        metrics: Sequence[str], *, n_convs: int = 2) -> pd.DataFrame:
    """Does this backend actually honour each rubric's schema? One real call per (metric, conv).

    Returns ``metric, n_ok, n_fail, mean_score``. A failure here is a hard stop:
    ``run_judge_scoring`` swallows per-call exceptions and simply skips the conversation, so a
    rubric this backend cannot satisfy would surface as *biased missingness* on that metric
    rather than as an error.
    """
    from questionnaires import QuestionnaireID
    name_to_qid = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
                   "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
                   "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
                   "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}
    client = init_judge_client(judge)
    rows = combined.head(n_convs)
    out = []
    for m in metrics:
        oks, vals = 0, []
        for _, row in rows.iterrows():
            rdf = await evaluate_conversation_with_judge(
                client, judge, row["conversation"], name_to_qid[m])
            if rdf is not None and not rdf.isnull().values.any():
                oks += 1
                num = rdf.select_dtypes("number")
                if len(num.columns):
                    vals.append(float(num.iloc[0].mean()))
        out.append({"metric": m, "n_ok": oks, "n_fail": len(rows) - oks,
                    "mean_score": round(float(np.mean(vals)), 3) if vals else None})
    return pd.DataFrame(out)


async def probe_discrimination(judge: JudgeSpec, combined: pd.DataFrame,
                               metrics: Sequence[str], model_a: str, model_b: str,
                               *, n_convs: int = 12) -> pd.DataFrame:
    """Can this grader tell two arms apart that the primary oracle separates clearly?

    THE gate that a schema check cannot replace. Pick two model states the primary oracle
    puts far apart (e.g. a Base rollout vs a late iteration); if the local grader returns
    the same mean for both, or returns near-zero variance across conversations, it is not a
    measuring instrument and the full sweep would only produce expensive noise.

    Returns ``metric, mean_a, mean_b, delta, sd_pooled, degenerate`` — ``degenerate`` is True
    when the pooled per-conversation SD is tiny relative to the rubric's own range, i.e. the
    judge is answering from a template rather than reading.
    """
    from questionnaires import QuestionnaireID
    name_to_qid = {"Q1": QuestionnaireID.Q1, "Q2": QuestionnaireID.Q2,
                   "WAI-SR": QuestionnaireID.WAI_SR, "CSQ-8": QuestionnaireID.CSQ8,
                   "MI-SAT": QuestionnaireID.MI_SAT, "MITI": QuestionnaireID.MITI,
                   "PCT": QuestionnaireID.PCT, "MICI": QuestionnaireID.MICI}
    client = init_judge_client(judge)

    async def _score(model_name: str, metric: str) -> List[float]:
        sub = combined[combined["Model"] == model_name].head(n_convs)
        tasks = [evaluate_conversation_with_judge(client, judge, r["conversation"],
                                                  name_to_qid[metric])
                 for _, r in sub.iterrows()]
        vals = []
        for rdf in await asyncio.gather(*tasks):
            if rdf is not None and not rdf.isnull().values.any():
                num = rdf.select_dtypes("number")
                if len(num.columns):
                    vals.append(float(num.iloc[0].mean()))
        return vals

    rows = []
    for m in metrics:
        a, b = await _score(model_a, m), await _score(model_b, m)
        if not a or not b:
            rows.append({"metric": m, "mean_a": None, "mean_b": None, "delta": None,
                         "sd_pooled": None, "degenerate": True})
            continue
        sd = float(np.std(np.array(a + b), ddof=1))
        rows.append({"metric": m, "n_a": len(a), "n_b": len(b),
                     "mean_a": round(float(np.mean(a)), 3),
                     "mean_b": round(float(np.mean(b)), 3),
                     "delta": round(float(np.mean(b) - np.mean(a)), 3),
                     "sd_pooled": round(sd, 4), "degenerate": bool(sd < 1e-3)})
    return pd.DataFrame(rows)
