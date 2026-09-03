"""fake_oracle_server.py -- an OpenAI-compatible endpoint that answers json_schema requests.

A TEST DOUBLE for the endpoint, not for the model. It does not read the transcript in any
meaningful sense; it returns schema-shaped answers under a chosen policy. That is exactly what
makes it useful: it exercises everything between Exp4 and the wire -- the request shape, guided
JSON, the validation ladder, aggregation, and the oracle-sanity gate -- with no vLLM, no GPU and
no Colab session.

Why it earns a place in the repo: the two questions that otherwise wait for Colab are "does our
oracle request actually carry a usable schema" and "does the degeneracy gate really fire". Both
are answerable here in under a second, and the second one cannot be answered any other way --
a real grader that HAPPENS to be healthy proves nothing about whether the gate would have caught
a bad one. Point ``--policy degenerate`` at oracle_sanity and it must exit nonzero.

It advertises ``roles.DEFAULT_ORACLE_MODEL`` on ``/v1/models`` by default, so every tool that
adopts the default binding by port (``generate_convs`` loopback adoption, ``smoke.py roles``
without ``--model``) sees the model it expects rather than
``adopt_if_running``'s "serves a different model" refusal. ``--model`` overrides it.

Policies:
  varied      scores vary with a hash of the transcript  -- a healthy-shaped grader
  degenerate  every score the same constant              -- schema-valid, zero variance: the
                                                            failure the sanity gate exists for
  short       fewer items than the rubric demands        -- must trip the length check, because
                                                            silently accepting it is biased
                                                            missingness on the headline metric
  prose       plain text instead of JSON                 -- must trip parse/retry

Beyond ``chat.completions`` it mimics the two vLLM-specific surfaces the tools lean on:

* ``GET /v1/models`` carries ``max_model_len`` per entry, as vLLM's model card does, so
  ``tools.oracle_sanity.prompt_length_report`` can read the served context length.
* ``POST /tokenize`` answers ``{"count", "max_model_len", "tokens"}`` like vLLM's endpoint --
  with a WHITESPACE token count, which is a stand-in for the wire shape, not for Gemma's
  tokenizer. A length report against this double proves the plumbing, never the numbers.

A plain (non-schema) request whose last user message says ``Reply with exactly: <X>`` is
answered with ``<X>`` and an honest ``usage.completion_tokens``, so ``smoke.py roles`` can run
its READY probe against the double offline. That is the wire contract being exercised, not a
model being tested.

Usage::

    python tools/fake_oracle_server.py varied &
    python tools/oracle_sanity.py --base-url http://127.0.0.1:8123/v1 --provider openai_compat

    python tools/fake_oracle_server.py degenerate --port 8124 --model google/gemma-4-E2B-it

⚠ Localhost only, no auth, no TLS. A test double, never a service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Sequence

# `python tools/fake_oracle_server.py` puts tools/ on sys.path, NOT code/, so `roles` would not
# resolve when this file runs as a script. The trainer notebooks already prepend code/.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from roles import DEFAULT_ORACLE_MODEL, ServeSpec  # noqa: E402

POLICIES = ("varied", "degenerate", "short", "prose")
DEFAULT_PORT = 8123

POLICY = "varied"
MODEL_ID = DEFAULT_ORACLE_MODEL
MAX_MODEL_LEN = ServeSpec(model=DEFAULT_ORACLE_MODEL).max_model_len
CALLS = []

_REPLY_EXACTLY_RE = re.compile(r"reply with exactly:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def _n_items(schema: dict) -> int:
    """Item count the schema demands, from minItems/maxItems on the scores array."""
    try:
        arr = schema["properties"]["scores"]
        return int(arr.get("minItems") or arr.get("maxItems") or 5)
    except Exception:
        return 5


def _bounds(schema: dict):
    try:
        it = schema["properties"]["scores"]["items"]
        return int(it.get("minimum", 1)), int(it.get("maximum", 5))
    except Exception:
        return 1, 5


def _qid(schema: dict) -> int:
    try:
        return int(schema["properties"]["questionnaire_id"]["enum"][0])
    except Exception:
        return 1


def _payload(schema: dict, prompt: str) -> str:
    n = _n_items(schema)
    lo, hi = _bounds(schema)
    qid = _qid(schema)

    if POLICY == "degenerate":
        scores = [4] * n
    elif POLICY == "short":
        scores = [3] * max(1, n - 2)
    else:
        # Deterministic but transcript-dependent, so different conversations really do differ.
        # Keyed on the TRANSCRIPT half of the prompt (after the rubric), which is what varies.
        tail = prompt[-1200:]
        h = hashlib.blake2b(tail.encode("utf-8"), digest_size=8).digest()
        scores = [lo + (h[i % len(h)] % (hi - lo + 1)) for i in range(n)]
    return json.dumps({"questionnaire_id": qid, "scores": scores})


def _plain_reply(messages: Sequence[dict]) -> str:
    """Answer a non-schema chat request: echo a 'Reply with exactly: X' probe, else prose."""
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            last_user = str(m.get("content") or "")
    match = _REPLY_EXACTLY_RE.search(last_user)
    if match:
        return match.group(1)
    return "I would rate this conversation quite highly overall."


def _count_tokens(text: str) -> int:
    """The double's 'tokenizer': whitespace pieces. A wire-shape stand-in, not a tokenizer."""
    return len(text.split())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fake",
                                       "max_model_len": MAX_MODEL_LEN}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        req = self._read_json()
        path = self.path.rstrip("/")

        if path.endswith("/tokenize"):
            # vLLM's shape: either a raw `prompt` or chat `messages` (template applied there;
            # here the messages are simply concatenated, which is the honest thing a double can do).
            if isinstance(req.get("messages"), list):
                text = "\n".join(str(m.get("content") or "") for m in req["messages"])
            else:
                text = str(req.get("prompt") or "")
            count = _count_tokens(text)
            self._send(200, {"count": count, "max_model_len": MAX_MODEL_LEN,
                             "tokens": list(range(count))})
            return

        CALLS.append(req)
        rf = req.get("response_format") or {}
        schema = (rf.get("json_schema") or {}).get("schema") or {}
        messages = req.get("messages", [])
        prompt = "".join(str(m.get("content") or "") for m in messages)

        if POLICY == "prose" or not schema:
            content = _plain_reply(messages) if not schema else \
                "I would rate this conversation quite highly overall."
        else:
            content = _payload(schema, prompt)

        n_prompt = _count_tokens(prompt)
        n_completion = _count_tokens(content)
        self._send(200, {
            "id": "chatcmpl-fake", "object": "chat.completion", "model": req.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                      "total_tokens": n_prompt + n_completion},
        })


def serve(port: int = DEFAULT_PORT, *, model: Optional[str] = None,
          max_model_len: Optional[int] = None):
    """Start the double on a daemon thread and return the ``HTTPServer`` (call ``shutdown()``).

    ``model`` / ``max_model_len`` override what ``/v1/models`` and ``/tokenize`` advertise; both
    are module globals because the handler class has no instance state per server.
    """
    global MODEL_ID, MAX_MODEL_LEN
    if model:
        MODEL_ID = str(model)
    if max_model_len:
        MAX_MODEL_LEN = int(max_model_len)
    srv = HTTPServer(("127.0.0.1", int(port)), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fake_oracle_server",
        description="OpenAI-compatible TEST DOUBLE for the oracle/patient endpoint (no model).",
    )
    parser.add_argument("policy", nargs="?", default="varied", choices=POLICIES,
                        help="how scores are shaped (default: varied)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"loopback port (default {DEFAULT_PORT})")
    parser.add_argument("--model", default=DEFAULT_ORACLE_MODEL,
                        help="model id advertised on /v1/models (default: roles.DEFAULT_ORACLE_MODEL)")
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                        help="context length advertised on /v1/models and /tokenize")
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    POLICY = _args.policy
    _srv = serve(_args.port, model=_args.model, max_model_len=_args.max_model_len)
    print(f"fake server on http://127.0.0.1:{_args.port}/v1 policy={POLICY} model={MODEL_ID} "
          f"max_model_len={MAX_MODEL_LEN}", flush=True)
    threading.Event().wait()
