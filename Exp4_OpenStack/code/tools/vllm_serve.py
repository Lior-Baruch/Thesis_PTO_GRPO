"""vllm_serve.py -- bring up, adopt, and keep alive the local server the open roles talk to.

Exp4's premise is that the oracle, the patient and the judge are open models served locally, so
an arm costs GPU-hours and nothing else. That premise rests on exactly one piece of process
management: an OpenAI-compatible endpoint must be listening on a known port before the first
patient turn is simulated, and it must still be listening eight hours later when the last
look-ahead rollout is scored. Everything downstream (``core/conversations.py``, ``core/oracle.py``,
the EDA's scoring pass) speaks plain OpenAI protocol -- ``chat.completions`` plus
``response_format={"type": "json_schema"}`` via guided decoding -- and does not care what is
behind the URL. This module is the only place that knows there is a subprocess at all.

Without it, three things break in ways that are annoying rather than loud:

* **Re-running a notebook cell.** ``serve_roles`` is called from cell 3 of both trainers, and
  cell 3 gets re-run constantly (a typo in cell 1, a Colab reconnect, a kernel restart that did
  not take the server with it). Blindly launching a second ``vllm serve`` on an occupied port
  produces a process that dies on bind and a caller that waits out a 900 s timeout for a port
  that was healthy the whole time. Hence :func:`adopt_if_running`: a healthy server already
  serving the right model is *adopted*, not duplicated, and one serving a DIFFERENT model is a
  hard error rather than a silent mis-grading of the entire run.
* **A server that dies mid-arm.** The oracle path would then see a burst of connection errors,
  exhaust its retries, and return ``None`` scores -- which ``core/reward.py`` turns into a
  ``min_success_ratio`` abort. :func:`ensure_alive` is called at every phase boundary and from
  that retry path so the common case (the server went away) is repaired instead of aborting the
  iteration. It gives up after ``max_restarts`` because a server that keeps dying is a config
  problem, and restarting forever would turn it into a mysterious slowdown.
* **Guessing at memory.** :func:`report_weights_gib` reads the real weight figure out of the
  vLLM startup log, so the Phase 1 gate records a measured number instead of the estimate.

Memory guidance (carried over from Exp3's ``local_server.py``, and still the thing that bites)
----------------------------------------------------------------------------------------------
``gpu_memory_utilization`` is a **PRE-ALLOCATION, not a ceiling that grows on demand**. vLLM
reserves that fraction of the card for weights plus KV pool at startup and never gives it back.
Three consequences:

* Scoring or grading on an otherwise-idle GPU wants it HIGH (0.85-ish): the KV pool is what buys
  request concurrency, and there is nothing else to starve.
* Sharing the card with a live trainer -- the Exp4 default -- wants it LOW (~0.25) and the server
  started **FIRST**. Training memory is the spiky side (GRPO's 128-completion generate, DPO's
  full-sequence logits over a 128k vocab), so it should get the slack, and it can only get the
  slack if the fixed reservation is already carved out and small.
* On the 12 GB local card an over-budget request does not raise ``OutOfMemoryError`` -- it
  hard-faults the GPU and takes the machine down with it. Do the arithmetic with
  :func:`estimate_vram_gib` before launching, which is what ``tools/smoke.py`` does.

The rubric-first prompt layout in ``questionnaires.py`` exists so vLLM's prefix cache reuses the
fixed instructions across every oracle call; prefix caching is on by default on current vLLM, and
older builds want ``extra_args=("--enable-prefix-caching",)`` on the :class:`~roles.ServeSpec`.

Typical use::

    from roles import make_binding
    from tools.vllm_serve import serve_roles, ensure_alive, report_weights_gib

    bindings, handles = serve_roles(bindings, base_port=8000, gpu_memory_utilization=0.25)
    for h in handles.values():
        print(h.model, report_weights_gib(h), "GiB of weights")
    ...
    for h in handles.values():          # at every phase boundary
        ensure_alive(h)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Dict, Iterator, List, Optional, Tuple, Union

from roles import RoleBinding, ServeSpec, model_tag, plan_servers

__all__ = [
    "DEFAULT_HOST",
    "ServerHandle",
    "base_url_for_port",
    "wait_until_ready",
    "start_server",
    "adopt_if_running",
    "serve_roles",
    "ensure_alive",
    "report_weights_gib",
    "serve",
    "estimate_vram_gib",
    "detect_total_vram_gib",
]

# 127.0.0.1 rather than "localhost": on Windows "localhost" can resolve to ::1 first, and a
# server bound only to IPv4 then costs every probe a failed connect before the retry. The URL
# built here is the one handed to the OpenAI SDK, so keep the two spellings from diverging by
# always going through base_url_for_port().
DEFAULT_HOST = "127.0.0.1"

# A readiness probe must be short: it runs in a poll loop, and a slow answer is a "not ready"
# either way.
_PROBE_TIMEOUT_S = 5.0

# report_weights_gib scans from the top of the log; the startup banner is the first few hundred
# lines, and a long-running server's log is mostly request noise after that.
_LOG_SCAN_MAX_LINES = 20000


# ---------------------------------------------------------------------------------------------
# HTTP probing helpers (stdlib only -- this module must be importable before torch and without
# the openai SDK, since it runs at notebook cell 3, ahead of every heavy import)
# ---------------------------------------------------------------------------------------------

def base_url_for_port(port: int, *, host: str = DEFAULT_HOST) -> str:
    """The OpenAI-compatible base URL a server on *port* answers at (``.../v1``)."""
    return f"http://{host}:{int(port)}/v1"


def _models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _get_json(url: str, *, timeout: float = _PROBE_TIMEOUT_S) -> Optional[dict]:
    """GET *url* and parse JSON, or ``None`` for any failure at all.

    Deliberately total: every caller here treats "did not answer with parseable JSON" and
    "answered with an error" the same way, and a probe must never raise into a poll loop.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _served_model_ids(payload: Optional[dict]) -> List[str]:
    """Model ids from an OpenAI ``/models`` payload; ``[]`` if the shape is unexpected."""
    if not payload:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            out.append(entry["id"])
    return out


def _model_ids_match(wanted: str, served: str) -> bool:
    """Is *served* the same model as *wanted*?

    Exact match, or equal basenames case-insensitively. The slack covers a server launched with
    ``--served-model-name gemma`` or a registry prefix, without being loose enough to call a
    Llama a Gemma -- and mistaking those is exactly the failure
    :func:`adopt_if_running` refuses to allow.
    """
    if wanted == served:
        return True
    return wanted.split("/")[-1].lower() == served.split("/")[-1].lower()


def _port_in_use(port: int, *, host: str = DEFAULT_HOST, timeout: float = 0.5) -> bool:
    """Is anything accepting TCP on *port*? Used to tell "free" from "still loading weights"."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port_release(port: int, *, timeout: float = 15.0,
                           poll_seconds: float = 0.5) -> bool:
    """Wait for *port* to stop accepting connections. Returns whether it did.

    A terminated vLLM does not always release its listening socket the instant ``wait()``
    returns (worker subprocesses can outlive the parent briefly). Without this, a restart in
    :func:`ensure_alive` would immediately trip ``start_server``'s occupied-port guard and turn
    a recoverable death into a raised exception mid-arm.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_in_use(port):
            return True
        time.sleep(poll_seconds)
    return not _port_in_use(port)


def _default_log_path(spec: ServeSpec, log_dir: Optional[str]) -> str:
    """Log path for *spec*. Port plus model tag, so two servers never share a file.

    ``model_tag`` is ``[A-Za-z0-9]`` only, which is what makes this a legal NTFS filename for a
    model id containing ``/`` and ``.``.
    """
    directory = log_dir or os.getcwd()
    return os.path.join(directory, f"vllm_{spec.port}_{model_tag(spec.model)}.log")


# ---------------------------------------------------------------------------------------------
# The handle
# ---------------------------------------------------------------------------------------------

@dataclass
class ServerHandle:
    """A running (or externally-managed) OpenAI-compatible server.

    ``process`` is ``None`` for an adopted server -- one this process did not launch. ``stop()``
    is then a no-op by design: a notebook cell re-run must not be able to kill the server the
    previous cell started and the running trainer is using.

    ``spec`` is kept so :func:`ensure_alive` can relaunch the *identical* configuration. Restart
    a server under a different ``gpu_memory_utilization`` or ``max_model_len`` and the arm's
    second half runs under different serving conditions than its first, with nothing in the run
    metadata to say so.
    """

    model: str
    base_url: str
    process: Optional[subprocess.Popen]
    log_path: Optional[str]
    spec: ServeSpec
    restarts: int = 0

    def stop(self, timeout: float = 30.0) -> None:
        """Terminate the server if this process owns it; no-op for an adopted handle."""
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def tail_log(self, n: int = 40) -> str:
        """Last *n* lines of the startup log, or a placeholder. Never raises.

        This is what turns "the server did not come up" into a diagnosable event: the reason
        (OOM during weight load, a bad flag, an occupied port) is always in these lines.
        """
        if not self.log_path or not os.path.exists(self.log_path):
            return "(no log)"
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-n:])
        except OSError as exc:
            return f"(could not read log: {exc})"

    def is_alive(self) -> bool:
        """Is this server answering ``GET {base_url}/models`` right now?

        Notes:
            "Answering right now" is not the same as "the process exists". A server still
            loading weights is not alive by this definition, which is the correct answer for
            every caller here -- they all want to know whether a request would succeed.
        """
        if self.process is not None and self.process.poll() is not None:
            return False
        return _get_json(_models_url(self.base_url)) is not None


# ---------------------------------------------------------------------------------------------
# Launch / readiness
# ---------------------------------------------------------------------------------------------

def wait_until_ready(base_url: str, *, timeout: float = 900.0,
                     process: Optional[subprocess.Popen] = None,
                     poll_seconds: float = 3.0) -> None:
    """Block until ``GET {base_url}/models`` answers, or raise.

    Args:
        base_url: the ``.../v1`` endpoint to poll.
        timeout: total seconds to wait. Weight load plus CUDA graph capture is minutes, not
            seconds, on a cold Colab runtime -- 900 is not paranoia.
        process: if given, watched between polls.
        poll_seconds: sleep between probes.

    Raises:
        RuntimeError: *process* exited before the port opened.
        TimeoutError: the deadline passed with the port still closed.

    Notes:
        The *process* watch is the reason this is not a bare poll loop. A server that dies during
        weight loading (OOM is the usual reason) would otherwise leave the caller waiting out the
        entire timeout for a port that is never going to open -- 15 minutes of silence for a
        failure that was known in 40 seconds.
    """
    deadline = time.time() + timeout
    url = _models_url(base_url)
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode} before becoming ready -- "
                f"check the log (OOM during weight load is the usual cause)")
        try:
            with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
        time.sleep(poll_seconds)
    raise TimeoutError(f"server at {base_url} not ready after {timeout:.0f}s ({last_err})")


def start_server(spec: ServeSpec, *, log_dir: Optional[str] = None, timeout: float = 900.0,
                 executable: str = "vllm") -> ServerHandle:
    """Launch ``vllm serve`` for *spec* and block until it answers; caller owns ``handle.stop()``.

    Args:
        spec: the full serving configuration. The command line is built from the dataclass and
            nothing else, so ``run_metadata.json`` recording the spec records the launch exactly.
        log_dir: directory for the server log (default: cwd). The path is printed, because the
            log is the only place a startup failure explains itself.
        timeout: passed to :func:`wait_until_ready`.
        executable: the server binary. Point it at another OpenAI-compatible server to swap
            backends without touching any caller.

    Raises:
        RuntimeError: the port is already occupied (use :func:`adopt_if_running` instead), the
            executable is missing, or the server exited during startup.
        TimeoutError: the server never answered.

    Notes:
        The notebook-friendly half of :func:`serve` -- a ``with`` block cannot span notebook
        cells, and the server has to outlive the cell that started it.
    """
    if shutil.which(executable) is None and not os.path.exists(executable):
        raise RuntimeError(
            f"server executable {executable!r} not found on PATH. Install vLLM "
            f"(pip install vllm) or pass executable= pointing at another "
            f"OpenAI-compatible server binary.")
    if _port_in_use(spec.port):
        raise RuntimeError(
            f"port {spec.port} is already occupied but did not adopt cleanly. Either another "
            f"model is served there (adopt_if_running says which), or a server is still loading "
            f"its weights -- wait for it and re-run, or pick another port.")

    base_url = base_url_for_port(spec.port)
    log_path = _default_log_path(spec, log_dir)
    cmd = [
        executable, "serve", spec.model,
        "--port", str(spec.port),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--max-model-len", str(spec.max_model_len),
        "--dtype", str(spec.dtype),
        *tuple(spec.extra_args or ()),
    ]
    print("[vllm_serve] launching:", " ".join(cmd))
    print("[vllm_serve] log:", log_path)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    finally:
        # The child holds its own duplicate of the handle, so the parent's copy is dead weight;
        # keeping it open leaks one descriptor per (re)start.
        log.close()

    handle = ServerHandle(model=spec.model, base_url=base_url, process=proc,
                          log_path=log_path, spec=spec)
    try:
        wait_until_ready(base_url, timeout=timeout, process=proc)
    except BaseException:
        print(handle.tail_log())
        handle.stop()
        raise
    print(f"[vllm_serve] ready: {spec.model} @ {base_url}")
    return handle


def adopt_if_running(spec: ServeSpec, *, log_dir: Optional[str] = None,
                     probe_timeout: float = _PROBE_TIMEOUT_S) -> Optional[ServerHandle]:
    """Adopt a server already listening on ``spec.port``, or ``None`` if the port is free.

    Args:
        spec: the configuration the caller *wants*. Only ``model`` and ``port`` are checked --
            a running server's ``gpu_memory_utilization`` cannot be read back over HTTP.
        log_dir: where a log from a previous launch of this spec would be, so
            :func:`report_weights_gib` still works on an adopted handle.
        probe_timeout: seconds for the single ``/models`` probe.

    Returns:
        A :class:`ServerHandle` with ``process=None`` (externally managed), or ``None``.

    Raises:
        RuntimeError: something answers on that port but serves a **different** model. That is
            an error, never an adoption: silently talking to the wrong grader would produce a
            complete, valid-looking, wrongly-scored arm -- the most expensive failure available.

    Notes:
        This is what makes :func:`serve_roles` idempotent across notebook cell re-runs and Colab
        session resumes. Re-running cell 3 must not try to bind an occupied port.
        A handle returned here has ``process=None``, so ``stop()`` on it does nothing by design.
    """
    base_url = base_url_for_port(spec.port)
    payload = _get_json(_models_url(base_url), timeout=probe_timeout)
    served = _served_model_ids(payload)
    if payload is None:
        return None
    if not served:
        # Answered /models with no entries: an OpenAI-compatible shell with nothing loaded.
        # Not adoptable, and not ours to reason about.
        return None
    if not any(_model_ids_match(spec.model, sid) for sid in served):
        raise RuntimeError(
            f"port {spec.port} already serves {served!r} but this run needs {spec.model!r}. "
            f"Refusing to adopt: every score would be graded by the wrong model. Stop that "
            f"server, or give this spec a different port.")

    log_path = _default_log_path(spec, log_dir)
    return ServerHandle(model=spec.model, base_url=base_url, process=None,
                        log_path=log_path if os.path.exists(log_path) else None, spec=spec)


# ---------------------------------------------------------------------------------------------
# The one call the notebooks make
# ---------------------------------------------------------------------------------------------

def serve_roles(bindings: Dict[str, RoleBinding], *, base_port: int = 8000,
                log_dir: Optional[str] = None, timeout: float = 900.0,
                executable: str = "vllm",
                **spec_kw) -> Tuple[Dict[str, RoleBinding], Dict[str, ServerHandle]]:
    """Plan, start-or-adopt, and wire the servers every open role needs.

    Args:
        bindings: role name -> :class:`~roles.RoleBinding` (``{"oracle": ..., "patient": ...,
            "judge": ...}``).
        base_port: first port to allocate; ``plan_servers`` assigns from here.
        log_dir: directory for server logs (default: cwd).
        timeout: readiness timeout per server.
        executable: server binary.
        **spec_kw: forwarded to ``plan_servers`` and thus onto every
            :class:`~roles.ServeSpec` (``gpu_memory_utilization``, ``max_model_len``, ``dtype``,
            ``extra_args``).

    Returns:
        ``(bindings, handles)``. *bindings* is a NEW dict with ``base_url`` filled in for every
        ``openai_compat`` role (``RoleBinding`` is frozen, so this is ``dataclasses.replace``);
        API-provider bindings pass through untouched and identical. *handles* is keyed by
        **model id, not role name** -- ``plan_servers`` dedupes by model, so patient + oracle +
        judge on one Gemma share exactly one handle.

    Notes:
        With no ``openai_compat`` binding this starts nothing and returns the bindings unchanged
        with an empty handles dict, so the all-API arm keeps the same call in the notebook.

        Idempotent: a healthy server already on the port serving the right model is adopted. A
        port that is occupied but not yet answering is treated as "the same server, still
        loading" and waited out -- that is the cell-re-run-during-startup case, and launching a
        second process there would only produce a bind failure.

        Call this at cell 3, BEFORE any torch import, so the server's fixed pre-allocation is
        carved out before the trainer starts claiming the spiky remainder.
    """
    specs = plan_servers(bindings, base_port=base_port, **spec_kw)
    if not specs:
        print("[vllm_serve] no openai_compat bindings -- nothing to serve")
        return dict(bindings), {}

    handles: Dict[str, ServerHandle] = {}
    for spec in specs:
        handle = adopt_if_running(spec, log_dir=log_dir)
        if handle is not None:
            print(f"[vllm_serve] adopted {spec.model} @ {handle.base_url} "
                  f"(externally managed -- not stopping it on exit)")
        elif _port_in_use(spec.port):
            print(f"[vllm_serve] port {spec.port} occupied but not answering yet -- "
                  f"assuming a server still loading weights; waiting")
            wait_until_ready(base_url_for_port(spec.port), timeout=timeout)
            handle = adopt_if_running(spec, log_dir=log_dir)
            if handle is None:
                raise RuntimeError(
                    f"port {spec.port} opened but reports no model; cannot adopt or launch "
                    f"{spec.model!r} there.")
            print(f"[vllm_serve] adopted {spec.model} @ {handle.base_url} after its startup")
        else:
            handle = start_server(spec, log_dir=log_dir, timeout=timeout, executable=executable)
        handles[spec.model] = handle

    wired: Dict[str, RoleBinding] = {}
    for role, binding in bindings.items():
        handle = handles.get(binding.model) if binding.is_local else None
        if handle is None:
            wired[role] = binding                       # API provider, or an explicit remote URL
            continue
        if binding.base_url and binding.base_url != handle.base_url:
            print(f"[vllm_serve] WARNING: role {role!r} carried base_url "
                  f"{binding.base_url!r}; replacing with {handle.base_url!r}")
        wired[role] = replace(binding, base_url=handle.base_url)

    for role, binding in wired.items():
        where = binding.base_url or binding.provider
        print(f"[vllm_serve]   {role:<8} {binding.model}  ->  {where}")
    return wired, handles


def ensure_alive(handle: ServerHandle, *, max_restarts: int = 3,
                 grace_seconds: float = 30.0, timeout: float = 900.0) -> ServerHandle:
    """Verify the server is answering; relaunch the same spec if it is not.

    Args:
        handle: the handle to check. **Repaired in place** and returned, so a handle stored in
            the dict from :func:`serve_roles` stays valid for every other holder.
        max_restarts: how many relaunches this handle is allowed over its whole life.
        grace_seconds: how long a still-running-but-silent server is given to answer before it
            is treated as wedged and killed.
        timeout: readiness timeout for the relaunch.

    Returns:
        The same :class:`ServerHandle` object, alive.

    Raises:
        RuntimeError: the restart budget is exhausted. A server that keeps dying is a config
            problem -- almost always ``gpu_memory_utilization`` colliding with the trainer's
            peak -- and restarting forever would present it as a mysterious slowdown instead.

    Notes:
        Called at every phase boundary (generate / build / train) and from the oracle and
        patient retry paths on a burst of connection errors, which is the symptom of exactly
        this failure. Cheap when healthy: one HTTP probe.
    """
    if handle.is_alive():
        return handle

    if handle.process is not None and handle.process.poll() is None:
        # Alive but silent: give it the grace window before killing something that may just be
        # busy or still capturing CUDA graphs.
        try:
            wait_until_ready(handle.base_url, timeout=grace_seconds,
                             process=handle.process, poll_seconds=2.0)
            return handle
        except (TimeoutError, RuntimeError):
            print(f"[vllm_serve] {handle.model} @ {handle.base_url} is running but not "
                  f"answering after {grace_seconds:.0f}s -- restarting it")

    if handle.restarts >= max_restarts:
        raise RuntimeError(
            f"server {handle.model} @ {handle.base_url} died again after {handle.restarts} "
            f"restart(s) (limit {max_restarts}). This is a configuration problem, not a "
            f"transient -- check gpu_memory_utilization against the trainer's peak. Log tail:\n"
            f"{handle.tail_log()}")

    handle.stop()
    _wait_for_port_release(handle.spec.port)
    log_dir = os.path.dirname(handle.log_path) if handle.log_path else None
    fresh = start_server(handle.spec, log_dir=log_dir, timeout=timeout)
    handle.process = fresh.process
    handle.log_path = fresh.log_path
    handle.base_url = fresh.base_url
    handle.restarts += 1
    print(f"[vllm_serve] restarted {handle.model} (restart {handle.restarts}/{max_restarts})")
    return handle


@contextmanager
def serve(spec: Union[ServeSpec, str], **kwargs) -> Iterator[ServerHandle]:
    """Context-manager form for scripts -- stops the server on exit.

    Accepts a :class:`~roles.ServeSpec` or a bare model id plus ``ServeSpec`` field kwargs
    (``port``, ``gpu_memory_utilization``, ``max_model_len``, ``dtype``, ``extra_args``); the
    rest go to :func:`start_server` (``log_dir``, ``timeout``, ``executable``).

    Notes:
        An already-running server is adopted, and an adopted handle's ``stop()`` is a no-op, so
        a ``with`` block will not kill a server it did not start. For notebooks use
        :func:`serve_roles` or :func:`start_server` instead -- a ``with`` block cannot span
        cells, and the server must outlive the cell.
    """
    if isinstance(spec, str):
        spec_fields = {f.name for f in fields(ServeSpec)} - {"model"}
        spec_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in spec_fields}
        spec = ServeSpec(model=spec, **spec_kw)
    handle = adopt_if_running(spec, log_dir=kwargs.get("log_dir"))
    if handle is None:
        handle = start_server(spec, **kwargs)
    try:
        yield handle
    finally:
        handle.stop()
        if handle.process is not None:
            print("[vllm_serve] server stopped")


# ---------------------------------------------------------------------------------------------
# Measurement: what the server actually took, and what a spec will claim
# ---------------------------------------------------------------------------------------------

# vLLM has spelled this line several ways across versions; try them in order of specificity and
# fall back to a loose one. A wording change must degrade to None, never to a wrong number and
# never to an exception -- this is called at a gate, not in a hot path.
_WEIGHT_PATTERNS = (
    re.compile(r"model weights take\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"loading model weights took\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"model loading took\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"model weights[^0-9\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
)


def report_weights_gib(handle: ServerHandle) -> Optional[float]:
    """Model-weight memory in GiB, parsed from the vLLM startup log; ``None`` if not found.

    Args:
        handle: a handle whose ``log_path`` points at the launch log. An adopted handle only has
            one if a previous launch in the same ``log_dir`` left it there.

    Returns:
        The figure as printed, or ``None``.

    Notes:
        This exists to replace the estimated "~3 GB for Gemma-4-E2B bf16" with a MEASURED number
        at the Phase 1 gate, because the whole 40 GB Colab budget is built on that estimate.
        Older vLLM labels the same quantity "GB" while computing bytes/2**30; the difference is
        labelling, not value, so no unit conversion is applied.
        Never raises: an unreadable log or a reworded line returns ``None``, and the caller
        reports "unknown" rather than failing a gate on a log-format change.
    """
    path = handle.log_path if isinstance(handle, ServerHandle) else None
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= _LOG_SCAN_MAX_LINES:
                    break
                for pattern in _WEIGHT_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        try:
                            return float(match.group(1))
                        except (TypeError, ValueError):
                            return None
    except OSError:
        return None
    return None


def detect_total_vram_gib(device: int = 0) -> Optional[float]:
    """Total VRAM on *device* in GiB, or ``None`` if it cannot be determined.

    Notes:
        ``nvidia-smi`` is tried FIRST on purpose: reading the figure through torch initializes a
        CUDA context in this process (a few hundred MB), and the point of the caller is to
        decide whether there is room *before* claiming any. torch is only imported as a
        fallback, and only lazily -- this module is imported at notebook cell 3, ahead of the
        deliberate ``import trl`` -> ``import torch`` order the local Blackwell card requires.
    """
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=False)
            lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
            if 0 <= device < len(lines):
                return float(lines[device]) / 1024.0          # nvidia-smi reports MiB
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        import torch                                          # noqa: PLC0415 - lazy on purpose
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            return float(props.total_memory) / (1024.0 ** 3)
    except Exception:
        return None
    return None


def estimate_vram_gib(spec: ServeSpec, *, total_gib: Optional[float] = None,
                      device: int = 0) -> float:
    """GiB this spec will PRE-ALLOCATE on *device* (``gpu_memory_utilization`` x total VRAM).

    Args:
        spec: the serving configuration about to be launched.
        total_gib: override the detected card size (for arithmetic about another machine).
        device: CUDA device index.

    Returns:
        The reservation in GiB. Not an estimate of peak use -- vLLM claims this at startup and
        holds it, so it is exactly the number to subtract from the budget before deciding
        whether a trainer or a conversation batch also fits.

    Raises:
        RuntimeError: no GPU could be detected and no ``total_gib`` was given. Returning 0.0
            there would silently defeat ``tools/smoke.py``'s refuse-before-allocating guard,
            and on the 12 GB local card an over-budget request reboots the machine rather than
            raising an ``OutOfMemoryError`` anyone could catch.
    """
    total = total_gib if total_gib is not None else detect_total_vram_gib(device)
    if total is None:
        raise RuntimeError(
            "could not determine total VRAM (no nvidia-smi, no CUDA torch). Pass total_gib= "
            "to do the arithmetic anyway.")
    return round(float(spec.gpu_memory_utilization) * float(total), 3)
