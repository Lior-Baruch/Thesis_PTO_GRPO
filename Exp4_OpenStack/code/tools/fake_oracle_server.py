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

Policies:
  varied      scores vary with a hash of the transcript  -- a healthy-shaped grader
  degenerate  every score the same constant              -- schema-valid, zero variance: the
                                                            failure the sanity gate exists for
  short       fewer items than the rubric demands        -- must trip the length check, because
                                                            silently accepting it is biased
                                                            missingness on the headline metric
  prose       plain text instead of JSON                 -- must trip parse/retry

Usage::

    python tools/fake_oracle_server.py varied &
    python tools/oracle_sanity.py --base-url http://127.0.0.1:8123/v1         --model google/gemma-4-E2B-it --provider openai_compat

⚠ Localhost only, no auth, no TLS. A test double, never a service.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

POLICY = "varied"
CALLS = []


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

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": "google/gemma-4-E2B-it", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        CALLS.append(req)

        rf = req.get("response_format") or {}
        schema = (rf.get("json_schema") or {}).get("schema") or {}
        prompt = "".join(m.get("content", "") for m in req.get("messages", []))

        if POLICY == "prose" or not schema:
            content = "I would rate this conversation quite highly overall."
        else:
            content = _payload(schema, prompt)

        self._send(200, {
            "id": "chatcmpl-fake", "object": "chat.completion", "model": req.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })


def serve(port=8123):
    srv = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


if __name__ == "__main__":
    import sys
    POLICY = sys.argv[1] if len(sys.argv) > 1 else "varied"
    s = serve()
    print(f"fake server on http://127.0.0.1:8123/v1 policy={POLICY}")
    threading.Event().wait()
