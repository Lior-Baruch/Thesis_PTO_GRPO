"""vllm_serve.py -- bring up, adopt, and keep alive the local server the open roles talk to.

Exp4's premise is that the oracle, the patient and the judge are open models served locally, so
an arm costs GPU-hours and nothing else. That premise rests on exactly one piece of process
management: an OpenAI-compatible endpoint must be listening on a known port before the first
patient turn is simulated, and it must still be listening eight hours later when the last
look-ahead rollout is scored. Everything downstream (``core/conversations.py``, ``core/oracle.py``,
the EDA's scoring pass) speaks plain OpenAI protocol -- ``chat.completions`` plus
``response_format={"type": "json_schema"}`` via guided decoding -- and does not care what is
behind the URL. This module is the only place that knows there is a subprocess at all.

Without it, four things break in ways that are annoying rather than loud:

* **Re-running a notebook cell.** ``serve_roles`` is called from cell 3 of both trainers, and
  cell 3 gets re-run constantly (a typo in cell 1, a Colab reconnect, a kernel restart that did
  not take the server with it). Blindly launching a second ``vllm serve`` on an occupied port
  produces a process that dies on bind and a caller that waits out the whole readiness timeout
  for a port that was healthy the whole time. Hence :func:`adopt_if_running`: a healthy server
  already serving the right model is *adopted*, not duplicated, and one serving a DIFFERENT
  model is a hard error rather than a silent mis-grading of the entire run.
* **Re-running the cell while the server is still LOADING.** vLLM binds its port only after the
  engine is built -- minutes after launch on a cold runtime, longer with a ~15 GiB download in
  front of it. During that window the port is closed, so both the HTTP probe and the
  port-in-use check say "nothing there", and a naive re-run launches a SECOND ``vllm serve``
  that will claim a second pre-allocation (or die when the first one finally binds). Hence the
  module-level **registry** of servers this process launched (:func:`launched_servers`) and
  the POSIX ``pgrep`` fallback (:func:`find_loading_server`) for the kernel-restart case, where
  the registry is empty but the process is still there. Both are consulted before any launch.
* **A server that dies mid-arm.** The oracle path would then see a burst of connection errors,
  exhaust its retries, and return ``None`` scores -- which ``core/reward.py`` turns into a
  ``min_success_ratio`` abort. :func:`ensure_alive` is the repair, and it runs at the TRAINERS'
  phase boundaries, never inside ``core/``: GRPO probes through
  ``grpo_trainer.ensure_servers_alive`` before its generate phase, before its train phase and
  before the post-loop final eval (plus once at the top of the notebook loop); PTO probes at the
  top of its iteration loop, before the preference build, before the DPO update and before its
  final eval (``pto_trainer.ensure_servers_alive`` via the ``server_handles`` /
  ``client_factory`` kwargs on ``run_one_iteration`` / ``run_final_eval``). Nothing in
  ``core/conversations.py`` or ``core/oracle.py`` calls it -- their retry loops exhaust and
  report, and the phase that follows finds the dead server here. So a mid-phase death costs
  the rest of that phase, not the arm. It gives up after ``max_restarts`` because a server
  that keeps dying is a config problem, and restarting forever would turn it into a mysterious
  slowdown. It relaunches under the spec the server was ORIGINALLY started with -- recorded when
  this process launched it, recovered from the command line when it was adopted -- and refuses
  when that is unknowable: ``Run_Eval`` adopts the trainer's server with its own ``util 0.85``
  spec, and relaunching under that would hand the eval's reservation to a card the trainer is
  still using.
* **Guessing at memory.** :func:`report_weights_gib` and :func:`report_kv_cache_tokens` read
  the real weight and KV-pool figures out of the vLLM startup log, so the Phase 1 gate records
  measured numbers instead of the estimate.

Memory guidance (carried over from Exp3's ``local_server.py``, and still the thing that bites)
----------------------------------------------------------------------------------------------
``gpu_memory_utilization`` is a **PRE-ALLOCATION, not a ceiling that grows on demand**. vLLM
reserves that fraction of the card for weights plus KV pool at startup and never gives it back.
Three consequences:

* Scoring or grading on an otherwise-idle GPU wants it HIGH (0.85-ish): the KV pool is what buys
  request concurrency, and there is nothing else to starve.
* Sharing the card with a live trainer -- the Exp4 default -- wants the sanctioned per-model
  fraction from ``roles.DEFAULT_SERVE_UTIL`` (E4B 0.50, E2B 0.35) and the server started
  **FIRST**. Training memory is the spiky side (GRPO's 128-completion generate, DPO's
  full-sequence logits over a 128k vocab), so it should get the slack, and it can only get the
  slack if the fixed reservation is already carved out.
* On the 12 GB local card an over-budget request does not raise ``OutOfMemoryError`` -- it
  hard-faults the GPU and takes the machine down with it. Do the arithmetic with
  :func:`estimate_vram_gib` before launching, which is what ``tools/smoke.py`` does.

The rubric-first prompt layout in ``questionnaires.py`` exists so vLLM's prefix cache reuses the
fixed instructions across every oracle call; prefix caching is on by default on current vLLM, and
older builds want ``extra_args=("--enable-prefix-caching",)`` on the :class:`~roles.ServeSpec`.

Host assumptions: none. Nothing here imports Colab or torch; the log directory defaults to
``./_vllm_logs`` under the current working directory, and the process discovery fallback uses
``pgrep`` where it exists (any Linux box -- Colab or a GPU server over SSH) and degrades to the
in-process registry alone where it does not (Windows).

Typical use::

    from roles import default_serve_util, make_binding
    from tools.vllm_serve import serve_roles, ensure_alive, report_weights_gib

    bindings, handles = serve_roles(bindings, base_port=8000,
                                    gpu_memory_utilization=default_serve_util(model))
    for h in handles.values():
        print(h.model, report_weights_gib(h), "GiB of weights")
    ...
    for h in handles.values():          # at a trainer phase boundary (generate / build / train)
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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from roles import RoleBinding, ServeSpec, model_tag, plan_servers

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_LOG_DIRNAME",
    "DEFAULT_READY_TIMEOUT",
    "SPEC_SOURCES",
    "ServerHandle",
    "base_url_for_port",
    "served_max_model_len",
    "wait_until_ready",
    "launched_servers",
    "spec_from_cmdline",
    "find_loading_server",
    "start_server",
    "adopt_if_running",
    "serve_roles",
    "ensure_alive",
    "report_weights_gib",
    "report_kv_cache_tokens",
    "report_max_concurrency",
    "serve",
    "estimate_vram_gib",
    "detect_total_vram_gib",
]

# 127.0.0.1 rather than "localhost": on Windows "localhost" can resolve to ::1 first, and a
# server bound only to IPv4 then costs every probe a failed connect before the retry. The URL
# built here is the one handed to the OpenAI SDK, so keep the two spellings from diverging by
# always going through base_url_for_port().
DEFAULT_HOST = "127.0.0.1"

#: Where server logs go when no ``log_dir`` is given: a subdirectory of the cwd rather than the
#: cwd itself, so a GPU server's working tree does not fill with ``vllm_8000_*.log`` files and
#: the adopt path knows one place to look for a previous launch's log.
DEFAULT_LOG_DIRNAME = "_vllm_logs"

#: Readiness timeout in seconds. 1800, not 900: a cold Colab runtime downloads the ~15 GiB E4B
#: checkpoint BEFORE it loads it, and the download alone has been observed past 10 minutes.
#: :func:`wait_until_ready` also extends its own deadline while the log shows progress, so this
#: is the budget for a server that has gone SILENT, not for one that is visibly working.
DEFAULT_READY_TIMEOUT = 1800.0

#: How a :class:`ServerHandle` came to know its ``spec`` -- see the attribute docs there.
SPEC_SOURCES = ("launched", "cmdline", "requested")

# A readiness probe must be short: it runs in a poll loop, and a slow answer is a "not ready"
# either way.
_PROBE_TIMEOUT_S = 5.0

# report_* scan from the top of the log; the startup banner is the first few hundred lines, and
# a long-running server's log is mostly request noise after that.
_LOG_SCAN_MAX_LINES = 20000

# wait_until_ready's progress heuristic: a log line matching this is "the server is still doing
# something" (a weight download, a shard load, CUDA-graph capture, torch.compile). Deliberately
# broad -- a false positive only delays a timeout, a false negative only shortens one.
_PROGRESS_RE = re.compile(
    r"(download|fetch|loading|loaded|\d+%|it/s|weights|shard|captur|compil|warm)", re.IGNORECASE
)

# Hard cap on how far progress lines may push the readiness deadline, as a multiple of the
# timeout. A log that never stops growing but never opens the port is not a loading server.
_PROGRESS_TOTAL_MULTIPLIER = 4.0

# How many bytes of log tail to inspect per poll for progress markers.
_PROGRESS_TAIL_BYTES = 4096

#: In-process registry of servers THIS process launched, keyed by port. Consulted before every
#: launch so a re-run of the serve cell during the multi-minute engine build cannot start a
#: second process on the same port (the port is unbound until the engine is up, so neither the
#: HTTP probe nor a socket test can see the first one). Entries whose process has exited are
#: evicted on sight.
_LAUNCHED: Dict[int, "ServerHandle"] = {}


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


def _served_model_entries(payload: Optional[dict]) -> List[dict]:
    """Model entries from an OpenAI ``/models`` payload; ``[]`` if the shape is unexpected."""
    if not payload:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [entry for entry in data if isinstance(entry, dict) and isinstance(entry.get("id"), str)]


def _served_model_ids(payload: Optional[dict]) -> List[str]:
    """Model ids from an OpenAI ``/models`` payload; ``[]`` if the shape is unexpected."""
    return [entry["id"] for entry in _served_model_entries(payload)]


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


def served_max_model_len(base_url: str, *, model: Optional[str] = None,
                         timeout: float = _PROBE_TIMEOUT_S) -> Optional[int]:
    """The context length the server at *base_url* advertises for *model*, or ``None``.

    vLLM's model card carries ``max_model_len``; the OpenAI API's does not, and neither does
    every compatible server, so ``None`` means "not advertised", never "unlimited". With
    ``model=None`` the first entry is used (the Exp4 server serves exactly one model).
    """
    payload = _get_json(_models_url(base_url), timeout=timeout)
    for entry in _served_model_entries(payload):
        if model is not None and not _model_ids_match(model, entry["id"]):
            continue
        raw = entry.get("max_model_len")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _port_in_use(port: int, *, host: str = DEFAULT_HOST, timeout: float = 0.5) -> bool:
    """Is anything accepting TCP on *port*? Used to tell "free" from "bound, not answering"."""
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
    model id containing ``/`` and ``.``. The default directory is ``./_vllm_logs`` (see
    :data:`DEFAULT_LOG_DIRNAME`).
    """
    directory = log_dir or os.path.join(os.getcwd(), DEFAULT_LOG_DIRNAME)
    return os.path.join(directory, f"vllm_{spec.port}_{model_tag(spec.model)}.log")


def _pid_alive(pid: Optional[int]) -> Optional[bool]:
    """Is *pid* still running? ``None`` when the host cannot say (Windows, or no pid).

    POSIX only: ``os.kill(pid, 0)`` is a permission/existence probe, not a signal. ``None``
    rather than ``True`` on Windows so a caller never treats "could not check" as "alive".
    """
    if pid is None or os.name != "posix":
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


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
    metadata to say so. ``spec_source`` records how trustworthy that spec is:

    * ``"launched"`` -- this process built the command line from it. Authoritative.
    * ``"cmdline"`` -- recovered from the running process's argv (``pgrep``), see
      :func:`spec_from_cmdline`. Authoritative for the fields vLLM was given; anything the
      launcher left at vLLM's own default is filled from :class:`~roles.ServeSpec`'s defaults.
    * ``"requested"`` -- the spec the ADOPTING caller wanted; the server was only seen over
      HTTP and may have been started under anything. The one field the server does report,
      ``max_model_len`` (vLLM's model card), is replaced by the advertised value when it
      differs, with a printed WARNING; ``gpu_memory_utilization`` and ``dtype`` stay the
      caller's wish. :func:`ensure_alive` refuses to relaunch such a handle, because
      ``Run_Eval`` adopts the trainer's ``util 0.50`` server with its own ``util 0.85`` spec
      and a relaunch under that would take the trainer's memory.

    ``pid`` is the server's process id when known without owning it (recovered by ``pgrep``);
    ``executable`` is what a relaunch runs.
    """

    model: str
    base_url: str
    process: Optional[subprocess.Popen]
    log_path: Optional[str]
    spec: ServeSpec
    restarts: int = 0
    spec_source: str = "launched"
    pid: Optional[int] = None
    executable: str = "vllm"

    def __post_init__(self) -> None:
        if self.spec_source not in SPEC_SOURCES:
            raise ValueError(f"spec_source must be one of {SPEC_SOURCES}, got {self.spec_source!r}")
        if self.pid is None and self.process is not None:
            self.pid = self.process.pid

    @property
    def owns_process(self) -> bool:
        """True when this handle can stop/relaunch the server (it launched it)."""
        return self.process is not None

    def stop(self, timeout: float = 30.0) -> None:
        """Terminate the server if this process owns it; no-op for an adopted handle."""
        if _LAUNCHED.get(self.spec.port) is self:
            _LAUNCHED.pop(self.spec.port, None)
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


def launched_servers() -> Dict[int, ServerHandle]:
    """``{port: handle}`` for every server THIS process launched and that is still running.

    A copy of the registry with dead entries evicted, so a caller can inspect it without
    holding a reference into module state. Empty after a kernel restart -- that is what the
    ``pgrep`` fallback in :func:`find_loading_server` covers.
    """
    for port, handle in list(_LAUNCHED.items()):
        if handle.process is None or handle.process.poll() is not None:
            _LAUNCHED.pop(port, None)
    return dict(_LAUNCHED)


# ---------------------------------------------------------------------------------------------
# Discovering a server that has not bound its port yet
# ---------------------------------------------------------------------------------------------

def spec_from_cmdline(argv: Sequence[str], *, fallback: ServeSpec) -> ServeSpec:
    """Rebuild the :class:`~roles.ServeSpec` a ``vllm serve ...`` command line encodes.

    Args:
        argv: The process's argument vector (``["vllm", "serve", "<model>", "--port", ...]``,
            or with a ``python -m vllm ...`` / interpreter prefix in front).
        fallback: Supplies every field the command line does not mention -- normally the spec
            the caller wanted, so its model id is only replaced when argv names one.

    Returns:
        A spec whose ``model`` / ``port`` / ``gpu_memory_utilization`` / ``max_model_len`` /
        ``dtype`` come from argv where present; unrecognised flags land in ``extra_args`` in
        their original order.

    Notes:
        The inverse of :func:`start_server`'s command builder, so a relaunch from a recovered
        spec reproduces the original launch. vLLM accepts both ``--flag value`` and
        ``--flag=value``; both are read. A flag vLLM defaulted (absent from argv) is filled from
        *fallback*, which is the one place the recovered spec can differ from the real launch --
        :class:`~roles.ServeSpec`'s defaults mirror the notebooks' explicit values, so in
        practice they agree.
    """
    args = [str(a) for a in argv]
    try:
        start = args.index("serve") + 1
    except ValueError:
        start = 0
    rest = args[start:]

    model = fallback.model
    if rest and not rest[0].startswith("-"):
        model = rest[0]
        rest = rest[1:]

    known = {
        "--port": ("port", int),
        "--gpu-memory-utilization": ("gpu_memory_utilization", float),
        "--max-model-len": ("max_model_len", int),
        "--dtype": ("dtype", str),
    }
    values: Dict[str, object] = {}
    extra: List[str] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        flag, eq, inline = token.partition("=")
        if flag in known:
            field, cast = known[flag]
            raw = inline if eq else (rest[i + 1] if i + 1 < len(rest) else None)
            if raw is not None:
                try:
                    values[field] = cast(raw)
                except (TypeError, ValueError):
                    extra.extend([token] if eq else rest[i:i + 2])
            i += 1 if eq else 2
            continue
        extra.append(token)
        i += 1

    return replace(fallback, model=model, extra_args=tuple(extra), **values)


def _pgrep_vllm_serve(port: int) -> Optional[Tuple[int, List[str]]]:
    """``(pid, argv)`` of a ``vllm serve`` process bound for *port*, via ``pgrep``; else ``None``.

    POSIX only (``pgrep`` from procps -- present on Colab and on any Linux GPU server). On a
    host without it this returns ``None`` and the caller falls back to the registry alone.
    A process whose command line carries no ``--port`` is taken to be on vLLM's default 8000.
    """
    exe = shutil.which("pgrep")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-af", "vllm"], capture_output=True, text=True,
                             timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (out.stdout or "").splitlines():
        parts = line.strip().split()
        if len(parts) < 2 or "serve" not in parts[1:]:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        argv = parts[1:]
        listed: Optional[int] = None
        for i, token in enumerate(argv):
            if token == "--port" and i + 1 < len(argv):
                listed = _safe_int(argv[i + 1])
            elif token.startswith("--port="):
                listed = _safe_int(token.split("=", 1)[1])
        if (listed if listed is not None else 8000) == int(port):
            return pid, argv
    return None


def _safe_int(raw: str) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _registry_hit(spec: ServeSpec) -> Optional[ServerHandle]:
    """The live handle this process launched on ``spec.port``, or ``None``; raises on a model clash."""
    handle = launched_servers().get(spec.port)
    if handle is None:
        return None
    if not _model_ids_match(spec.model, handle.model):
        raise RuntimeError(
            f"port {spec.port} is held by a server THIS process launched for {handle.model!r}, "
            f"but this spec wants {spec.model!r}. Stop it (handle.stop()) or use another port.")
    return handle


def find_loading_server(spec: ServeSpec, *, log_dir: Optional[str] = None) -> Optional[ServerHandle]:
    """A server for *spec* that exists but has not bound its port yet, or ``None``.

    Two sources, in order: the in-process registry (a server this process launched, whose
    handle owns the process and is returned as-is), then ``pgrep`` for a ``vllm serve`` on
    ``spec.port`` started by someone else -- a previous kernel, a shell -- returned as an
    adopted handle (``process=None``) with its spec recovered from the command line
    (``spec_source="cmdline"``) and its ``pid`` recorded so readiness polling can fast-fail if
    it dies.

    Raises:
        RuntimeError: a process is loading a DIFFERENT model on that port. Never adopted.

    Notes:
        This does not wait; the caller decides (``serve_roles`` waits, ``start_server`` waits,
        a probe may just report). Neither source can see a server on another machine, and the
        ``pgrep`` half is absent on Windows, where the registry alone stands.
    """
    hit = _registry_hit(spec)
    if hit is not None:
        return hit

    found = _pgrep_vllm_serve(spec.port)
    if found is None:
        return None
    pid, argv = found
    recovered = spec_from_cmdline(argv, fallback=spec)
    if not _model_ids_match(spec.model, recovered.model):
        raise RuntimeError(
            f"pid {pid} is starting `vllm serve {recovered.model}` on port {spec.port}, but this "
            f"run needs {spec.model!r}. Refusing to adopt or to launch beside it: kill it "
            f"(`kill {pid}`) or give this spec a different port.")
    log_path = _default_log_path(recovered, log_dir)
    return ServerHandle(model=recovered.model, base_url=base_url_for_port(spec.port),
                        process=None, log_path=log_path if os.path.exists(log_path) else None,
                        spec=recovered, spec_source="cmdline", pid=pid)


# ---------------------------------------------------------------------------------------------
# Launch / readiness
# ---------------------------------------------------------------------------------------------

def _log_progress_probe(log_path: Optional[str], last_size: int) -> Tuple[int, bool]:
    """``(size_now, progressed)``: did the log grow with loading/download markers since *last_size*?"""
    if not log_path:
        return last_size, False
    try:
        size = os.path.getsize(log_path)
    except OSError:
        return last_size, False
    if size <= last_size:
        return size, False
    try:
        with open(log_path, "rb") as fh:
            fh.seek(max(last_size, size - _PROGRESS_TAIL_BYTES))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return size, False
    return size, bool(_PROGRESS_RE.search(tail))


def wait_until_ready(base_url: str, *, timeout: float = DEFAULT_READY_TIMEOUT,
                     process: Optional[subprocess.Popen] = None,
                     poll_seconds: float = 3.0,
                     log_path: Optional[str] = None,
                     pid: Optional[int] = None) -> None:
    """Block until ``GET {base_url}/models`` answers, or raise.

    Args:
        base_url: the ``.../v1`` endpoint to poll.
        timeout: seconds of SILENCE to tolerate (default :data:`DEFAULT_READY_TIMEOUT`). With a
            *log_path*, every poll that finds new download/loading lines in the log pushes the
            deadline back out to ``now + timeout``, capped at ``4 x timeout`` from the start --
            so a cold runtime that is visibly pulling 15 GiB is not timed out mid-download,
            while a server that has stopped logging still fails within *timeout*.
        process: if given, watched between polls.
        poll_seconds: sleep between probes.
        log_path: the server's log, for the progress heuristic above. Optional.
        pid: a process id to watch when *process* is not owned (an adopted, still-loading
            server found by ``pgrep``). POSIX only; ignored where liveness cannot be checked.

    Raises:
        RuntimeError: *process* exited (or *pid* vanished) before the port opened.
        TimeoutError: the deadline passed with the port still closed.

    Notes:
        The *process* watch is the reason this is not a bare poll loop. A server that dies during
        weight loading (OOM is the usual reason) would otherwise leave the caller waiting out the
        entire timeout for a port that is never going to open -- 30 minutes of silence for a
        failure that was known in 40 seconds.

        The progress heuristic is a heuristic: :data:`_PROGRESS_RE` matches the words vLLM and
        huggingface_hub use for downloads, shard loads, CUDA-graph capture and compilation. A
        false positive costs at most ``4 x timeout`` of waiting; a false negative costs nothing
        that the plain timeout did not already cost.
    """
    started = time.time()
    deadline = started + timeout
    hard_cap = started + timeout * _PROGRESS_TOTAL_MULTIPLIER
    url = _models_url(base_url)
    last_err: Optional[BaseException] = None
    log_size = 0
    extended = False
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode} before becoming ready -- "
                f"check the log (OOM during weight load is the usual cause)")
        if process is None and _pid_alive(pid) is False:
            raise RuntimeError(
                f"server process {pid} disappeared before its port opened -- check its log "
                f"(OOM during weight load is the usual cause)")
        try:
            with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
        log_size, progressed = _log_progress_probe(log_path, log_size)
        if progressed:
            new_deadline = min(time.time() + timeout, hard_cap)
            if new_deadline > deadline:
                deadline = new_deadline
                if not extended:
                    print(f"[vllm_serve] log shows loading progress; readiness deadline extends "
                          f"while it continues (hard cap {timeout * _PROGRESS_TOTAL_MULTIPLIER:.0f}s)")
                    extended = True
        time.sleep(poll_seconds)
    waited = time.time() - started
    why = (f"hard cap of {timeout * _PROGRESS_TOTAL_MULTIPLIER:.0f}s reached while the log kept "
           f"showing progress" if extended and waited >= timeout * _PROGRESS_TOTAL_MULTIPLIER
           else f"{timeout:.0f}s without progress")
    raise TimeoutError(f"server at {base_url} not ready after {waited:.0f}s ({why}; {last_err})")


def start_server(spec: ServeSpec, *, log_dir: Optional[str] = None,
                 timeout: float = DEFAULT_READY_TIMEOUT,
                 executable: str = "vllm") -> ServerHandle:
    """Launch ``vllm serve`` for *spec* and block until it answers; caller owns ``handle.stop()``.

    Args:
        spec: the full serving configuration. The command line is built from the dataclass and
            nothing else, so ``run_metadata.json`` recording the spec records the launch exactly.
        log_dir: directory for the server log (default: ``./_vllm_logs``). The path is printed,
            because the log is the only place a startup failure explains itself.
        timeout: passed to :func:`wait_until_ready`.
        executable: the server binary. Point it at another OpenAI-compatible server to swap
            backends without touching any caller.

    Returns:
        The launched server's handle -- or, when a server for this spec is ALREADY loading on
        the port (this process's registry, or ``pgrep``), that server's handle once it is
        ready. A second launch there would only produce a bind failure or a second
        pre-allocation, so "start" means "make sure one is coming up".

    Raises:
        RuntimeError: the port is occupied by something that is not a loading server for this
            spec (use :func:`adopt_if_running`), a different model is loading there, the
            executable is missing, or the server exited during startup.
        TimeoutError: the server never answered.

    Notes:
        The notebook-friendly half of :func:`serve` -- a ``with`` block cannot span notebook
        cells, and the server has to outlive the cell that started it.
    """
    loading = find_loading_server(spec, log_dir=log_dir)
    if loading is not None:
        print(f"[vllm_serve] a server for {spec.model} is already loading on port {spec.port} "
              f"({'this process' if loading.owns_process else f'pid {loading.pid}'}); "
              f"waiting for it instead of launching a second one")
        wait_until_ready(loading.base_url, timeout=timeout, process=loading.process,
                         log_path=loading.log_path, pid=loading.pid)
        print(f"[vllm_serve] ready: {loading.model} @ {loading.base_url}")
        return loading

    if shutil.which(executable) is None and not os.path.exists(executable):
        raise RuntimeError(
            f"server executable {executable!r} not found on PATH. Install vLLM "
            f"(pip install vllm) or pass executable= pointing at another "
            f"OpenAI-compatible server binary.")
    if _port_in_use(spec.port):
        raise RuntimeError(
            f"port {spec.port} is already occupied but did not adopt cleanly. Either another "
            f"model is served there (adopt_if_running says which), or something that is not a "
            f"vLLM server holds the port -- free it, or pick another port.")

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
                          log_path=log_path, spec=spec, spec_source="launched",
                          executable=executable)
    _LAUNCHED[spec.port] = handle          # registered BEFORE the wait: a re-run mid-load finds it
    try:
        wait_until_ready(base_url, timeout=timeout, process=proc, log_path=log_path)
    except BaseException:
        print(handle.tail_log())
        handle.stop()
        raise
    print(f"[vllm_serve] ready: {spec.model} @ {base_url}")
    return handle


def _reconcile_served_max_model_len(spec: ServeSpec, base_url: str, *, source: str,
                                    probe_timeout: float = _PROBE_TIMEOUT_S) -> ServeSpec:
    """*spec* with ``max_model_len`` replaced by what the server at *base_url* ADVERTISES.

    Only a running server knows the context length it actually allocated KV cache for, and
    vLLM's model card (``/v1/models`` -> ``max_model_len``) says so. An adopted handle built
    from the REQUESTED spec (or from a command line that never spelled ``--max-model-len``)
    would otherwise carry the adopter's wish -- 16384 -- into ``run_metadata.json`` and into
    ``ensure_alive``'s relaunch while the server was serving 8192, and the prompt-length gate
    would be checking against a cap the server never had. A server that advertises nothing
    (the OpenAI API, another compatible server) leaves the spec as it is.
    """
    served = served_max_model_len(base_url, model=spec.model, timeout=probe_timeout)
    if served is None or int(served) == int(spec.max_model_len):
        return spec
    print(f"[vllm_serve] WARNING: the server at {base_url} advertises max_model_len {served} "
          f"for {spec.model}, not the {spec.max_model_len} this spec {source}; adopting the "
          f"SERVED value -- a relaunch reproduces what is running, and the prompt-length gate "
          f"checks against the real cap")
    return replace(spec, max_model_len=int(served))


def adopt_if_running(spec: ServeSpec, *, log_dir: Optional[str] = None,
                     probe_timeout: float = _PROBE_TIMEOUT_S) -> Optional[ServerHandle]:
    """Adopt a server already answering on ``spec.port``, or ``None`` if nothing answers.

    Args:
        spec: the configuration the caller *wants*. Only ``model`` and ``port`` are checked
            against the server; the rest of a running server's configuration cannot be read
            back over HTTP -- see the ``spec_source`` note below.
        log_dir: where a log from a previous launch of this spec would be, so
            :func:`report_weights_gib` still works on an adopted handle.
        probe_timeout: seconds for the single ``/models`` probe.

    Returns:
        A :class:`ServerHandle`, or ``None``. Three provenances, best first: the handle this
        process launched (from the registry -- it owns the process, so ``stop()`` and
        ``ensure_alive`` work fully); an adopted handle whose spec was recovered from the
        process's command line via ``pgrep`` (``spec_source="cmdline"``); or an adopted handle
        carrying the REQUESTED spec (``spec_source="requested"``) when the server could only be
        seen over HTTP. ``process`` is ``None`` for the last two, so ``stop()`` is a no-op.
        For both adopted forms ``spec.max_model_len`` is the value the server ADVERTISES on
        ``/v1/models`` when that differs from the requested / recovered one (printed as a
        WARNING) -- the adopter's wish is not what is running.

    Raises:
        RuntimeError: something answers on that port but serves a **different** model. That is
            an error, never an adoption: silently talking to the wrong grader would produce a
            complete, valid-looking, wrongly-scored arm -- the most expensive failure available.

    Notes:
        This is what makes :func:`serve_roles` idempotent across notebook cell re-runs and Colab
        session resumes. Re-running cell 3 must not try to bind an occupied port.
        A server that exists but has not bound its port yet is NOT found here -- that is
        :func:`find_loading_server`'s job, which ``serve_roles`` calls next.
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

    # Our own server (same kernel): the registry handle owns the process -- hand it back rather
    # than a process-less twin, so a re-run of the serve cell keeps a handle that can restart it.
    own = _registry_hit(spec)
    if own is not None:
        return own

    # Someone else's process on this machine: recover the REAL launch spec from its argv so a
    # relaunch by ensure_alive reproduces it instead of the adopter's wish.
    found = _pgrep_vllm_serve(spec.port)
    if found is not None:
        pid, argv = found
        recovered = spec_from_cmdline(argv, fallback=spec)
        recovered = _reconcile_served_max_model_len(
            recovered, base_url, source="recovered from its command line (or defaulted)",
            probe_timeout=probe_timeout)
        log_path = _default_log_path(recovered, log_dir)
        return ServerHandle(model=spec.model, base_url=base_url, process=None,
                            log_path=log_path if os.path.exists(log_path) else None,
                            spec=recovered, spec_source="cmdline", pid=pid)

    # Seen over HTTP only: the spec is what the ADOPTER asked for, except for the one field the
    # server itself reports. The rest (util, dtype) stays a wish, which is why spec_source says so.
    adopted = _reconcile_served_max_model_len(spec, base_url, source="requested",
                                              probe_timeout=probe_timeout)
    log_path = _default_log_path(adopted, log_dir)
    return ServerHandle(model=spec.model, base_url=base_url, process=None,
                        log_path=log_path if os.path.exists(log_path) else None, spec=adopted,
                        spec_source="requested")


# ---------------------------------------------------------------------------------------------
# The one call the notebooks make
# ---------------------------------------------------------------------------------------------

def _describe_measurements(handle: ServerHandle) -> str:
    """``weights 14.9 GiB, KV cache 123,456 tokens (7.5 x 16384)`` -- or what could not be read."""
    weights = report_weights_gib(handle)
    kv = report_kv_cache_tokens(handle)
    parts = [f"weights {weights:.2f} GiB" if weights is not None else "weights: not in log"]
    if kv is not None:
        parts.append(f"KV cache {kv:,} tokens (~{kv / max(1, handle.spec.max_model_len):.1f} x "
                     f"max_model_len {handle.spec.max_model_len})")
    else:
        parts.append("KV cache: not in log")
    conc = report_max_concurrency(handle)
    if conc is not None:
        parts.append(f"max concurrency {conc:.1f}x")
    if not handle.log_path:
        parts.append("(adopted without a log; figures unavailable)")
    return ", ".join(parts)


def serve_roles(bindings: Dict[str, RoleBinding], *, base_port: int = 8000,
                log_dir: Optional[str] = None, timeout: float = DEFAULT_READY_TIMEOUT,
                executable: str = "vllm",
                **spec_kw) -> Tuple[Dict[str, RoleBinding], Dict[str, ServerHandle]]:
    """Plan, start-or-adopt, and wire the servers every open role needs.

    Args:
        bindings: role name -> :class:`~roles.RoleBinding` (``{"oracle": ..., "patient": ...,
            "judge": ...}``).
        base_port: first port to allocate; ``plan_servers`` assigns from here.
        log_dir: directory for server logs (default: ``./_vllm_logs``).
        timeout: readiness timeout per server (silence budget; see :func:`wait_until_ready`).
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

        Idempotent, in three layers: a healthy server already on the port serving the right
        model is adopted; a server that is still LOADING (port unbound) is found through the
        registry or ``pgrep`` and waited for; a port that is bound but not yet answering is
        treated as "the same server, still starting" and waited out. Only when all three come
        up empty is a process launched -- so re-running this cell at any point in a startup
        never produces a second ``vllm serve``.

        Call this at cell 3, BEFORE any torch import, so the server's fixed pre-allocation is
        carved out before the trainer starts claiming the spiky remainder. The summary printed
        at the end carries the MEASURED weights and KV-pool figures from the startup log; the
        Phase 1 gate should record those, not the estimate.
    """
    specs = plan_servers(bindings, base_port=base_port, **spec_kw)
    if not specs:
        print("[vllm_serve] no openai_compat bindings -- nothing to serve")
        return dict(bindings), {}

    handles: Dict[str, ServerHandle] = {}
    for spec in specs:
        handle = adopt_if_running(spec, log_dir=log_dir)
        if handle is not None:
            owner = ("this process" if handle.owns_process
                     else f"externally managed, spec {handle.spec_source}")
            print(f"[vllm_serve] adopted {spec.model} @ {handle.base_url} ({owner}"
                  f"{'' if handle.owns_process else ' -- not stopping it on exit'})")
        else:
            loading = find_loading_server(spec, log_dir=log_dir)
            if loading is not None:
                print(f"[vllm_serve] {spec.model} is still loading on port {spec.port} "
                      f"({'this process' if loading.owns_process else f'pid {loading.pid}'}); "
                      f"waiting for it rather than launching a second server")
                wait_until_ready(loading.base_url, timeout=timeout, process=loading.process,
                                 log_path=loading.log_path, pid=loading.pid)
                handle = loading
                print(f"[vllm_serve] adopted {spec.model} @ {handle.base_url} after its startup")
            elif _port_in_use(spec.port):
                print(f"[vllm_serve] port {spec.port} bound but not answering yet -- "
                      f"assuming a server still starting; waiting")
                wait_until_ready(base_url_for_port(spec.port), timeout=timeout,
                                 log_path=_default_log_path(spec, log_dir))
                handle = adopt_if_running(spec, log_dir=log_dir)
                if handle is None:
                    raise RuntimeError(
                        f"port {spec.port} opened but reports no model; cannot adopt or launch "
                        f"{spec.model!r} there.")
                print(f"[vllm_serve] adopted {spec.model} @ {handle.base_url} after its startup")
            else:
                handle = start_server(spec, log_dir=log_dir, timeout=timeout,
                                      executable=executable)
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
    for model, handle in handles.items():
        # Cross-check the cap the handle carries against what the server ADVERTISES, whatever
        # the provenance: a launched server may have clamped the request to the model's own
        # limit, and an adopted one may never have been asked for this value at all.
        advertised = served_max_model_len(handle.base_url, model=handle.model)
        if advertised is None:
            cap_note = "server advertises no max_model_len"
        elif int(advertised) == int(handle.spec.max_model_len):
            cap_note = "server agrees"
        else:
            cap_note = (f"WARNING: server advertises max_model_len {advertised} -- the spec's "
                        f"{handle.spec.max_model_len} is NOT what is running")
            print(f"[vllm_serve] {cap_note} for {model} @ {handle.base_url}")
        print(f"[vllm_serve]   {model}: util {handle.spec.gpu_memory_utilization} "
              f"max_model_len {handle.spec.max_model_len} [{handle.spec_source}; {cap_note}] -- "
              f"{_describe_measurements(handle)}")
    return wired, handles


def ensure_alive(handle: ServerHandle, *, max_restarts: int = 3,
                 grace_seconds: float = 30.0,
                 timeout: float = DEFAULT_READY_TIMEOUT) -> ServerHandle:
    """Verify the server is answering; relaunch the ORIGINAL spec if it is not.

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
        RuntimeError: the restart budget is exhausted (a server that keeps dying is a config
            problem -- almost always ``gpu_memory_utilization`` colliding with the trainer's
            peak -- and restarting forever would present it as a mysterious slowdown); an
            ADOPTED server is wedged (its process is still there, or something still holds
            the port, but it does not answer) and cannot be killed from here; or an adopted
            server is gone and its launch spec is unknowable (``spec_source="requested"``),
            in which case relaunching under the adopter's spec is refused -- see
            :class:`ServerHandle`.

    Notes:
        **Who calls this.** The trainers, at their phase boundaries -- nothing in ``core/``
        does, and the oracle / patient retry loops do not either (they exhaust their retries
        and report; the next boundary finds the dead server here). GRPO:
        ``grpo_trainer.ensure_servers_alive`` before the generate phase, before the train
        phase and before the post-loop final eval, plus the notebook's own probe at the top of
        each loop iteration. PTO: the top of its iteration loop, before the preference build,
        before the DPO update and before the final eval (``pto_trainer.ensure_servers_alive``,
        reached through the ``server_handles`` / ``client_factory`` kwargs). Cheap when
        healthy: one HTTP probe.

        **How an ADOPTED handle is diagnosed.** ``stop()`` is a no-op on it, so the only
        repair available is a relaunch, and a relaunch beside a server that is still there
        dies on bind (or claims a second pre-allocation). The decision is therefore made on
        the PROCESS first and the port second: ``_pid_alive(pid) is True`` means the process
        exists and is merely not answering -- wedged, raise -- whatever a connect to the port
        says, because a saturated accept backlog makes ``_port_in_use``'s connect time out
        and read as "free". Only when the pid is known dead or unknowable (Windows, no pid)
        AND the port is free is the server treated as gone and relaunched.
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

    pid_state: Optional[bool] = None
    if handle.process is None:
        # An adopted server gets the same grace window (a busy server can miss one 5 s probe),
        # watched by pid where the host can: wait_until_ready fast-fails when the pid vanishes.
        try:
            wait_until_ready(handle.base_url, timeout=grace_seconds, pid=handle.pid,
                             poll_seconds=2.0)
            return handle
        except TimeoutError:
            pid_state = _pid_alive(handle.pid)
        except RuntimeError:
            pid_state = False            # the pid disappeared during the grace window

    if handle.restarts >= max_restarts:
        raise RuntimeError(
            f"server {handle.model} @ {handle.base_url} died again after {handle.restarts} "
            f"restart(s) (limit {max_restarts}). This is a configuration problem, not a "
            f"transient -- check gpu_memory_utilization against the trainer's peak. Log tail:\n"
            f"{handle.tail_log()}")

    if handle.process is None:
        port_held = _port_in_use(handle.spec.port)
        if pid_state is True or port_held:
            # An ADOPTED handle owns no process: stop() is a no-op by design, so a wedged
            # external server (its process still there, or its port still held, but not
            # answering -- e.g. one orphaned by a kernel restart, or one whose accept backlog
            # is saturated so that even the port probe times out) cannot be repaired from
            # here. Falling through would launch a second server that dies on bind, or wait
            # on a port that never releases and then raise start_server's misleading error.
            # Name the real situation and the real fix instead.
            who = f"pid {handle.pid}" if handle.pid is not None else "unknown pid"
            how = ("its process is still running" if pid_state is True
                   else f"it still holds port {handle.spec.port}")
            raise RuntimeError(
                f"server {handle.model} @ {handle.base_url} was ADOPTED (this process does not "
                f"own it; {who}) and is wedged: {how} but it stopped answering (no reply "
                f"within {grace_seconds:.0f}s). It must be killed by hand -- e.g. "
                f"`pkill -f 'vllm serve'` (Linux) or by PID from `ss -ltnp` / Task Manager -- "
                f"then re-run this cell; serve_roles will start a fresh server it owns."
            )
        if handle.spec_source == "requested":
            raise RuntimeError(
                f"server {handle.model} @ {handle.base_url} was ADOPTED over HTTP and has gone "
                f"away, and the spec it was launched under is UNKNOWABLE from here (this handle "
                f"only carries the spec the adopter asked for: util "
                f"{handle.spec.gpu_memory_utilization}, max_model_len {handle.spec.max_model_len}). "
                f"Refusing to relaunch under that: an eval pass adopting the trainer's server "
                f"would otherwise restart it with the eval's reservation and take the trainer's "
                f"memory. Re-run the cell that started the server (serve_roles will launch a "
                f"fresh one it owns under the intended spec), or start it by hand."
            )
        gone = ("its process is gone" if pid_state is False
                else "its process cannot be checked on this host and its port is free")
        print(f"[vllm_serve] adopted server {handle.model} (pid {handle.pid}) -- {gone}; "
              f"relaunching under the spec recovered from its command line "
              f"(util {handle.spec.gpu_memory_utilization}, max_model_len "
              f"{handle.spec.max_model_len})")

    handle.stop()
    _wait_for_port_release(handle.spec.port)
    log_dir = os.path.dirname(handle.log_path) if handle.log_path else None
    fresh = start_server(handle.spec, log_dir=log_dir, timeout=timeout,
                         executable=handle.executable)
    handle.process = fresh.process
    handle.pid = fresh.pid
    handle.log_path = fresh.log_path
    handle.base_url = fresh.base_url
    handle.spec_source = "launched"
    handle.restarts += 1
    _LAUNCHED[handle.spec.port] = handle   # the shared handle, not start_server's twin
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
        a ``with`` block will not kill a server ANOTHER process started. A server THIS process
        launched earlier comes back as its owning handle (the registry), and the block stops
        it on exit -- the process that started it is the one giving it up. For notebooks use
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

# vLLM has spelled these lines several ways across versions; try them in order of specificity
# and fall back to a loose one. A wording change must degrade to None, never to a wrong number
# and never to an exception -- these are called at a gate, not in a hot path.
_WEIGHT_PATTERNS = (
    re.compile(r"model weights take\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"loading model weights took\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"model loading took\s+([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
    re.compile(r"model weights[^0-9\n]{0,40}?([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB)", re.IGNORECASE),
)

# The KV pool, as vLLM v1 reports it after profiling ("GPU KV cache size: 1,234,567 tokens").
# Only the explicit tokens line is read: the older "# GPU blocks: N" form needs the block size
# to convert, and a guessed block size would be a wrong number rather than None.
_KV_TOKEN_PATTERNS = (
    re.compile(r"GPU KV cache size:\s*([0-9][0-9,]*)\s*tokens", re.IGNORECASE),
    re.compile(r"KV cache size[^0-9\n]{0,20}?([0-9][0-9,]*)\s*tokens", re.IGNORECASE),
)

# "Maximum concurrency for 16,384 tokens per request: 7.52x" (vLLM v1, same banner).
_CONCURRENCY_PATTERNS = (
    re.compile(r"Maximum concurrency for [0-9,]+ tokens per request:\s*([0-9]+(?:\.[0-9]+)?)x",
               re.IGNORECASE),
)


def _scan_log(handle: ServerHandle, patterns: Sequence["re.Pattern[str]"]) -> Optional[str]:
    """First capture group of the first pattern that matches in the startup log; else ``None``."""
    path = handle.log_path if isinstance(handle, ServerHandle) else None
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= _LOG_SCAN_MAX_LINES:
                    break
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        return match.group(1)
    except OSError:
        return None
    return None


def report_weights_gib(handle: ServerHandle) -> Optional[float]:
    """Model-weight memory in GiB, parsed from the vLLM startup log; ``None`` if not found.

    Args:
        handle: a handle whose ``log_path`` points at the launch log. An adopted handle only has
            one if a previous launch in the same ``log_dir`` left it there.

    Returns:
        The figure as printed, or ``None``.

    Notes:
        This exists to replace the estimated checkpoint figure with a MEASURED number at the
        Phase 1 gate, because the whole VRAM budget is built on that estimate. Older vLLM
        labels the same quantity "GB" while computing bytes/2**30; the difference is
        labelling, not value, so no unit conversion is applied.
        Never raises: an unreadable log or a reworded line returns ``None``, and the caller
        reports "unknown" rather than failing a gate on a log-format change.
    """
    raw = _scan_log(handle, _WEIGHT_PATTERNS)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def report_kv_cache_tokens(handle: ServerHandle) -> Optional[int]:
    """KV-pool capacity in tokens, parsed from the vLLM startup log; ``None`` if not found.

    The sibling of :func:`report_weights_gib` for the OTHER half of the pre-allocation: after
    loading the weights vLLM profiles a forward pass and sizes its KV pool from what is left of
    ``gpu_memory_utilization x card``, then logs ``GPU KV cache size: N tokens``. Divided by
    ``max_model_len`` that is the number of worst-case sequences the pool holds at once -- the
    concurrency the oracle and patient roles actually get, which is the figure the Phase 1
    gate should record next to the weights. Never raises; ``None`` on any log-format change.
    """
    raw = _scan_log(handle, _KV_TOKEN_PATTERNS)
    try:
        return int(raw.replace(",", "")) if raw is not None else None
    except (TypeError, ValueError):
        return None


def report_max_concurrency(handle: ServerHandle) -> Optional[float]:
    """vLLM's own ``Maximum concurrency for <max_model_len> tokens per request`` figure, or ``None``."""
    raw = _scan_log(handle, _CONCURRENCY_PATTERNS)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
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
