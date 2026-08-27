"""roles.py -- which model plays each LLM role, and which servers that implies.

Exp4 runs three LLM roles besides the therapist policy, and every one of them is selectable:

- **patient** -- simulates the client the therapist talks to. Defines the TASK; swapping it
  changes the environment, not just the measurement, so nothing is comparable across a change.
- **oracle** -- grades the training reward. Swapping it changes what the policy optimizes, so
  arms trained under different oracles are not comparable either.
- **judge** -- grades the *eval* scores after the fact. A judge swap is safe and re-runnable,
  which is why the score lake partitions on ``judge=<tag>`` instead of forcing a re-train.

The whole point of Exp4 is that all three default to an **open model served locally**
(``google/gemma-4-E2B-it`` behind a vLLM OpenAI-compatible server), so a full arm costs $0 in
API. Exp3's binding constraint was the OpenAI bill, and on 2026-08-20 it stopped being
theoretical -- an organization spend cap killed two Colab sessions outright. Nothing in this
module is about saving money directly; it is about making the *role -> provider* edge a
first-class, named thing so that flipping one role back to a vendor API (to sanity-check the
open grader, say) is a one-line config change that automatically widens the arm's identity.

This module is **stdlib-only and import-light** on purpose. Both the trainer (``code/core/``)
and the read-only EDA import it, and the EDA must never pull in torch, openai or anthropic just
to learn what a model tag is. Provider SDKs are imported lazily inside :func:`make_client`.

What this module owns
---------------------
1. :class:`RoleBinding` -- the (provider, model, endpoint, timeout/retry policy, per-request
   extra body) tuple for one role.
2. :func:`model_tag` -- the short ``[A-Za-z0-9]`` tag a model contributes to an arm name.
   ``naming.py`` composes those tags into ``EXPERIMENT_NAME``; this module never builds names.
3. :func:`plan_servers` -- the *deduplicated* set of vLLM servers a binding table requires.
   ``tools/vllm_serve.py`` turns that plan into processes; the plan itself is pure data, so the
   EDA and the smoke tests can reason about it without starting anything.

Coming from Exp3? Three functions are GONE
------------------------------------------
``binding_suffix``, ``suffix_from_tags`` and ``assert_name_matches_roles`` do not exist here and
have no replacement. In Exp3 a role tag appeared in an arm name **only for a non-default
binding**, because ~50k already-written score CSVs depended on default-bound names staying
byte-identical. That conditional suffix is exactly what made ``assert_name_matches_roles``
necessary: someone could edit ``ORACLE_MODEL_ID`` in a notebook, leave the hand-typed
``EXPERIMENT_NAME`` alone, and silently write a differently-rewarded policy into the default
arm's folder -- where the resume logic would then report "already scored".

Exp4 has no legacy lake, so role tags are **always** encoded and ``EXPERIMENT_NAME`` is
**computed, never typed** (see ``naming.build_experiment_name``). The failure those three
functions guarded against cannot occur, and a guard for an impossible failure is just a place
for the two name-building paths to drift apart.

Notes on ``extra_body``
-----------------------
Gemma 4 ships configurable thinking modes. The oracle and patient roles must run with thinking
OFF: thinking tokens cost latency on every one of the ~10k oracle calls per iteration, and a
reasoning preamble in front of a schema-constrained response is a good way to lose the schema.
:func:`thinking_off_extra_body` is the per-request switch, and :func:`make_binding` attaches it
by default -- but ONLY for ``openai_compat``, because the real OpenAI API rejects unknown body
keys with a 400.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "DEFAULT_ORACLE_MODEL",
    "DEFAULT_PATIENT_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_THERAPIST_MODEL",
    "PROVIDERS",
    "THINKING_OFF_EXTRA_BODY_JSON",
    "RoleBinding",
    "ServeSpec",
    "ORACLE_DEFAULT",
    "PATIENT_DEFAULT",
    "JUDGE_DEFAULT",
    "default_bindings",
    "model_tag",
    "thinking_off_extra_body",
    "make_binding",
    "plan_servers",
    "make_client",
    "reset_client_cache",
]


# ---------------------------------------------------------------------------
# Defaults -- the open stack
# ---------------------------------------------------------------------------

#: Gemma-4-E4B-it: 8.0B raw / 4B-class effective parameters (per-layer embeddings),
#: **14.89 GiB bf16 checkpoint** (HF API, 2026-08-26 -- vLLM loads the raw checkpoint; do NOT
#: assume PLE offload), 128K context, ungated Apache 2.0. The E2B sibling is 5.12B raw /
#: 9.54 GiB bf16 -- the fallback if E4B is too slow or too tight on the shared card.
#: One server serves all three roles; they differ only in per-request sampling params.
DEFAULT_ORACLE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_PATIENT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_JUDGE_MODEL = "google/gemma-4-E4B-it"

#: The therapist POLICY default. Unlike the three roles above it is not served (it lives in-process
#: on the training GPU), but since 2026-08-27 it IS selectable per arm -- base vs Instruct -- and
#: its tag is encoded in every ``EXPERIMENT_NAME`` (the ``_Th<tag>`` field), so two therapist
#: variants can never share a folder. The default is the Instruct variant: it ships the official
#: Llama-3 chat template (single-special-token stopping, no ChatML self-play class), so it is the
#: primary grid; the template-less base is the ``_ThL1B`` alternate arm.
DEFAULT_THERAPIST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

#: The providers :func:`make_client` knows how to construct. ``openai_compat`` is any
#: OpenAI-compatible server (vLLM, llama.cpp, TGI) -- same call shape, including
#: ``response_format={"type": "json_schema"}`` via guided decoding, which is what lets the
#: oracle's retry/validate stack work unchanged against a local model.
PROVIDERS = ("openai_compat", "openai", "anthropic")

#: The per-request body that turns Gemma 4's thinking mode off.
#:
#: WARNING: the kwarg name ``enable_thinking`` is UNVERIFIED against Gemma 4's published chat
#: template -- it is the convention several other thinking-capable open models use. The Phase 1
#: smoke gate (``tools/smoke.py roles``) is what proves it: it asserts a completion comes back
#: with no thinking block. A wrong key must fail LOUDLY there. It will not fail at request time
#: on its own -- vLLM passes ``chat_template_kwargs`` straight into the Jinja render, where an
#: unrecognised name is simply an unused variable, so the request succeeds and every subsequent
#: call quietly burns reasoning tokens. Do not skip the gate.
THINKING_OFF_EXTRA_BODY_JSON = '{"chat_template_kwargs": {"enable_thinking": false}}'


# ---------------------------------------------------------------------------
# Model tags
# ---------------------------------------------------------------------------

# Curated short tags. Anything not listed falls back to `_slugify`, which is stable but uglier;
# add an entry here when a model becomes a regular arm so names stay readable. The two Gemma
# entries below are what `_slugify` would produce anyway -- they are spelled out because these
# are the tags that appear in every Exp4 folder name, and a reader should be able to find them
# by grepping for the literal rather than by simulating the slugifier in their head.
_MODEL_TAGS = {
    "google/gemma-4-E2B-it": "gemma4E2B",
    "google/gemma-4-E4B-it": "gemma4E4B",
    "gpt-4o-mini-2024-07-18": "gpt4m",
    "gpt-4o-mini": "gpt4m",
    "gpt-4o": "gpt4o",
    "claude-haiku-4-5": "haiku45",
    # The two therapist variants. Curated because they appear in EVERY Exp4 arm name (the
    # _Th<tag> field) and because the derived slugs would be unreadably long.
    "meta-llama/Llama-3.2-1B": "L1B",
    "meta-llama/Llama-3.2-1B-Instruct": "L1Bi",
}


def _slugify(model_id: str) -> str:
    """Vendor-stripped alphanumeric tag: ``google/gemma-4-E2B-it`` -> ``gemma4E2B``.

    Restricted to ``[A-Za-z0-9]`` so a tag can sit inside an ``EXPERIMENT_NAME`` without
    colliding with the ``_``-delimited field structure the arm-name regex depends on -- and so
    every arm name is a legal Windows path segment and a legal TensorBoard logdir.

    ``-Instruct`` is deliberately NOT stripped (an earlier revision stripped it): a base model
    and its Instruct sibling are different policies, and stripping the suffix slugged
    ``Llama-3.2-1B`` and ``Llama-3.2-1B-Instruct`` to the SAME tag -- two different-policy arms
    sharing one folder, exactly the collision the always-encoded tags exist to prevent. ``-it``
    is still stripped (Gemma's instruction-tuned variants are the only ones this project serves,
    so the suffix carries no contrast).
    """
    s = model_id.split("/")[-1]
    for suffix in ("-it", "-hf"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return "".join(ch for ch in s if ch.isalnum()) or "model"


def model_tag(model_id: str) -> str:
    """Short filesystem/regex-safe tag for a model id.

    Args:
        model_id: A full model identifier (``google/gemma-4-E2B-it``, ``gpt-4o-mini``).

    Returns:
        ``[A-Za-z0-9]``-only tag; ``"none"`` for an empty/None id.

    Notes:
        This is one half of a round-trip that nothing else enforces: ``naming.py`` writes the
        tag into an arm name and the EDA reads it back out. The mapping is deliberately
        many-to-one (``gpt-4o-mini`` and ``gpt-4o-mini-2024-07-18`` both tag as ``gpt4m``), so a
        tag identifies a *model family for naming purposes*, not an exact snapshot. The exact
        model id belongs in ``run_metadata.json``.
    """
    if not model_id:
        return "none"
    return _MODEL_TAGS.get(model_id) or _slugify(model_id)


# ---------------------------------------------------------------------------
# RoleBinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleBinding:
    """Provider + model + call policy for one LLM role.

    Frozen and hashable, because bindings are used as cache keys (see :func:`make_client`) and
    are stashed inside frozen config dataclasses (``OracleConfig``, ``LookaheadConfig``). That
    is also why ``extra_body`` is stored as a JSON **string** rather than a dict -- a dict field
    would make the dataclass unhashable and quietly break both.

    Attributes:
        model: Full model id as the provider spells it.
        provider: One of :data:`PROVIDERS`. ``openai_compat`` needs ``base_url``.
        base_url: Endpoint for ``openai_compat``. Normally left ``None`` at construction and
            filled in by ``tools.vllm_serve.serve_roles`` once the port is known.
        api_key_env: Overrides which environment variable supplies the key. Local servers
            usually need no key at all -- :func:`make_client` sends a placeholder.
        temperature: Optional per-role sampling default. ``None`` means the caller decides;
            the oracle passes 0.0 and the patient its own value regardless, so this is only a
            convenience for ad-hoc callers.
        request_timeout: **PER ATTEMPT**, in seconds. Exp3's patient call had no timeout at
            all, which is the defect this fixes: the ``openai`` SDK default is 600 s, so a
            single hung look-ahead patient call added ten minutes to an optimizer step, and
            exhausting a long budget freezes a simulation outright. Under
            ``scale_rewards="group"`` ONE frozen sim shifts the mean AND the std of its group
            of 8 -- the damage is not confined to the sample that failed. The safe shape is a
            SHORT per-attempt timeout times MANY retries, never a long total budget.
        max_retries: Attempts the *caller's* retry loop should make. Exp4 owns its own backoff
            (see ``core.conversations.generate_patient_response``); the SDK's internal retries
            are disabled in :func:`make_client` so they cannot multiply the wall-clock budget
            invisibly.
        extra_body_json: JSON object string merged into the request body. ``openai_compat``
            only -- the OpenAI API returns 400 on unknown body keys.
    """

    model: str
    provider: str = "openai_compat"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: Optional[float] = None
    request_timeout: float = 90.0
    max_retries: int = 8
    extra_body_json: Optional[str] = None

    @property
    def tag(self) -> str:
        """Short model tag, as it appears in an arm name."""
        return model_tag(self.model)

    @property
    def is_local(self) -> bool:
        """True when this role is served by a local OpenAI-compatible server."""
        return self.provider == "openai_compat"

    @property
    def extra_body(self) -> Optional[dict]:
        """Parsed :attr:`extra_body_json`, or ``None`` when there is nothing to send.

        An empty object ``{}`` collapses to ``None`` so a caller can pass the result straight
        through to the SDK: ``client.chat.completions.create(..., extra_body=b.extra_body)``
        with ``None`` meaning "omit", which is what the SDK expects.

        Raises:
            ValueError: if the string is not a JSON object. This is deliberately loud. The one
                thing that must never happen is a malformed thinking-off switch degrading to
                "send nothing", because that failure is invisible -- the run keeps working and
                every call silently pays for reasoning tokens.
        """
        raw = self.extra_body_json
        if not raw or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"RoleBinding.extra_body_json is not valid JSON for {self.provider}:{self.model}: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"RoleBinding.extra_body_json must encode a JSON object, got "
                f"{type(parsed).__name__} for {self.provider}:{self.model}"
            )
        return parsed or None


def thinking_off_extra_body() -> str:
    """The ``extra_body_json`` value that disables Gemma 4's thinking mode.

    Returns:
        ``'{"chat_template_kwargs": {"enable_thinking": false}}'``

    Notes:
        WARNING -- the kwarg name is UNVERIFIED against Gemma 4's actual chat template; see
        :data:`THINKING_OFF_EXTRA_BODY_JSON` for why a wrong key fails *silently* at request
        time and why the Phase 1 smoke gate is the thing that proves it. If the gate shows
        thinking is still on, change the key here: it is the single place it is spelled.

        ``openai_compat`` only. Never attach this to an ``openai`` or ``anthropic`` binding --
        vendor APIs 400 on unknown body keys, and that failure arrives on the first real call
        of a run, not at config time.
    """
    return THINKING_OFF_EXTRA_BODY_JSON


def make_binding(provider: str,
                 model: str,
                 *,
                 base_url: Optional[str] = None,
                 disable_thinking: bool = True,
                 **kw) -> RoleBinding:
    """Construct a :class:`RoleBinding`, attaching the thinking-off body where it belongs.

    Args:
        provider: One of :data:`PROVIDERS`.
        model: Full model id.
        base_url: Endpoint for ``openai_compat``; leave ``None`` to let ``serve_roles`` fill it.
        disable_thinking: When True (default) and ``provider == "openai_compat"``, set
            ``extra_body_json`` to :func:`thinking_off_extra_body`. Ignored for every other
            provider, because the OpenAI API returns 400 on unknown body keys.
        **kw: Any other :class:`RoleBinding` field (``temperature``, ``request_timeout``,
            ``max_retries``, ``api_key_env``, ``extra_body_json``).

    Returns:
        A frozen :class:`RoleBinding`.

    Raises:
        ValueError: on an unknown provider -- caught here rather than at the first API call,
            which on a Colab run is ~40 minutes of generation later.

    Notes:
        An explicitly passed ``extra_body_json=`` always wins and is passed through untouched
        for ANY provider: only the *automatic* attachment is gated on ``openai_compat``. If you
        deliberately send an extra body to a vendor API, you own the 400.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {PROVIDERS}")

    if "extra_body_json" not in kw and disable_thinking and provider == "openai_compat":
        kw["extra_body_json"] = thinking_off_extra_body()

    return RoleBinding(model=model, provider=provider, base_url=base_url, **kw)


#: Ready-made bindings for the three open-stack roles. ``base_url`` is filled in by
#: ``tools.vllm_serve.serve_roles`` once a port exists, so these are templates, not endpoints.
ORACLE_DEFAULT = make_binding("openai_compat", DEFAULT_ORACLE_MODEL)
PATIENT_DEFAULT = make_binding("openai_compat", DEFAULT_PATIENT_MODEL)
JUDGE_DEFAULT = make_binding("openai_compat", DEFAULT_JUDGE_MODEL)


def default_bindings() -> Dict[str, RoleBinding]:
    """A fresh ``{role: RoleBinding}`` table for the all-open default stack.

    Returns a NEW dict each call so a caller can mutate one role without editing a module-level
    object that every other caller shares. The bindings themselves are frozen; replace an entry
    rather than trying to edit it (``b = dataclasses.replace(b, temperature=0.0)``).
    """
    return {"oracle": ORACLE_DEFAULT, "patient": PATIENT_DEFAULT, "judge": JUDGE_DEFAULT}


# ---------------------------------------------------------------------------
# Server planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServeSpec:
    """Everything ``tools/vllm_serve.py`` needs to bring one vLLM server up.

    Pure data, deliberately: the plan can be computed, printed, asserted on and diffed by the
    smoke tests and the EDA without importing vllm, torch, or starting a process.

    Attributes:
        model: The model id to serve.
        port: Localhost port. Assigned deterministically by :func:`plan_servers`.
        gpu_memory_utilization: vLLM's share of the card. This is a **pre-allocation, not a
            growing ceiling** -- vLLM grabs the fraction up front and keeps it. When the server
            shares a card with a live trainer this wants to be as low as the WEIGHTS allow and
            the server must start FIRST, because training memory is the spiky side. Real
            checkpoint sizes (HF API, 2026-08-26): Gemma-4-E4B-it 14.89 GiB bf16 -> 0.50 of a
            40 GB A100 (20 GiB = weights + ~4-5 GiB KV pool); Gemma-4-E2B-it 9.54 GiB -> 0.35.
            The dataclass default (0.25) fits neither on its own -- the notebooks pass the
            model-derived value explicitly; the default exists for tests and planning.
        max_model_len: Context window to allocate KV cache for.
            ⚠ **Do NOT lower this to save memory.** Measured on the 192 real Exp3 PTO_LA0
            transcripts, the full oracle prompt (rubric + transcript) runs to 9,319 tokens for Q1
            and **10,042 for Q2**; at 8192 that is 1.0% / 2.1% of conversations that cannot be
            scored at all. Those are the LONGEST conversations, and session length varies by arm
            and by K, so the dropout would be arm-dependent -- a silent bias on the headline
            metric rather than an error. 16384 covers every observed prompt with ~60% headroom.
            Give memory back via ``gpu_memory_utilization`` instead.
        dtype: Weight dtype for the served model.
        extra_args: Extra CLI arguments appended verbatim to the ``vllm serve`` command.
    """

    model: str
    port: int = 8000
    gpu_memory_utilization: float = 0.25
    max_model_len: int = 16384
    dtype: str = "bfloat16"
    extra_args: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible endpoint this spec will serve.

        ``127.0.0.1`` rather than ``localhost`` on purpose: on Windows ``localhost`` can resolve
        to ``::1`` first while the server binds IPv4 only, which shows up as a connection
        refusal that looks like the server failed to start.
        """
        return f"http://127.0.0.1:{self.port}/v1"


def plan_servers(bindings: Dict[str, RoleBinding],
                 *,
                 base_port: int = 8000,
                 **spec_kw) -> List[ServeSpec]:
    """The minimal set of vLLM servers a binding table requires.

    Args:
        bindings: ``{role: RoleBinding}``, e.g. ``{"oracle": ..., "patient": ..., "judge": ...}``.
        base_port: First port to assign; subsequent servers get ``base_port + 1``, etc.
        **spec_kw: Applied to every :class:`ServeSpec` (``gpu_memory_utilization``,
            ``max_model_len``, ``dtype``, ``extra_args``).

    Returns:
        One :class:`ServeSpec` per DISTINCT ``openai_compat`` model, sorted by model id.
        Bindings whose provider is not ``openai_compat`` produce no spec -- a vendor API needs
        no process.

    Raises:
        ValueError: if ``spec_kw`` contains ``model`` or ``port``, which this function assigns.

    Notes:
        **Deduplication is by model id.** The default Exp4 stack binds oracle, patient and judge
        to the same Gemma, and that is ONE server, not three: a single vLLM instance serves
        every role, and the roles differ only in per-request sampling params. Starting three
        would triple the weight memory and split the prefix cache three ways -- and prefix
        caching is precisely what makes the rubric-first oracle prompt cheap.

        **Port assignment is deterministic** (sorted by model id) so a re-plan in a resumed
        session lands on exactly the same ports, which is what lets
        ``vllm_serve.adopt_if_running`` recognise and adopt a server that is already up instead
        of duplicating it.

        A binding that already carries a ``base_url`` still produces a spec. Handling "that
        endpoint is already serving this model" is ``adopt_if_running``'s job, and it decides by
        asking the port, which is more trustworthy than a config string.

        LIMITATION: because the key is the model id alone, two roles bound to the same model at
        two DIFFERENT ``base_url``s (a local server plus a remote one) collapse into one spec.
        Exp4 has no such configuration; if one appears, widen the key to ``(model, base_url)``.
    """
    for reserved in ("model", "port"):
        if reserved in spec_kw:
            raise ValueError(
                f"plan_servers assigns {reserved!r} itself; pass base_port= instead of {reserved!r}"
            )

    models = sorted({b.model for b in bindings.values() if b.is_local})
    return [ServeSpec(model=m, port=base_port + i, **spec_kw) for i, m in enumerate(models)]


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

# Clients are cached per binding: the oracle path would otherwise build one per scoring call,
# and an async client holds a connection pool worth reusing across the ~10k calls an iteration
# makes. `request_timeout` is part of the key because the SDK bakes the timeout into the client
# -- oracle and patient share model+endpoint in the default stack, so keying without it would
# silently hand one role the other's timeout. The final key element is the RUNNING LOOP's id
# (None outside a loop): pooled keep-alive connections cannot legally cross event loops, and
# run_async spawns a fresh loop per call. See make_client's Notes.
_CLIENT_CACHE: dict = {}


def make_client(binding: RoleBinding, *, api_key: Optional[str] = None):
    """Async client for *binding*. Cached per (provider, model, base_url, timeout, key, LOOP).

    Args:
        binding: The role binding to build a client for.
        api_key: Explicit key; wins over ``binding.api_key_env``, which wins over the
            provider's default environment variable.

    Returns:
        ``AsyncOpenAI`` for ``openai``/``openai_compat``, ``AsyncAnthropic`` for ``anthropic``.

    Raises:
        RuntimeError: no key could be resolved for a provider that needs one.
        ValueError: unknown provider.

    Notes:
        **The cache is keyed by the running event loop** (mirroring ``AsyncPrimitives``), and a
        same-binding entry from a DIFFERENT loop is evicted on sight. ``run_async`` spawns a
        fresh loop per call and TRL runs the reward coroutine on its own persistent loop, so a
        client object crossing loops carries keep-alive connections that were opened on a loop
        that no longer exists -- measured on the pinned openai/httpx stack, every such parked
        connection poisons exactly one call on the next loop with ``APIConnectionError``, which
        can eat a whole retry budget at a phase boundary. Async entry points therefore re-resolve
        their client via this function INSIDE the running coroutine rather than reusing a handle
        built elsewhere; construction is cheap and per-loop reuse is a dict hit. Callers outside
        any loop get the ``None`` loop key (probes, sync setup).

        An ``openai_compat`` server that wants no auth gets the ``"EMPTY"`` placeholder, because
        the OpenAI SDK refuses to construct without some key.

        The client is built with ``max_retries=0`` **on purpose**. Exp4 runs its own retry loop
        with its own backoff, and it sleeps that backoff OUTSIDE the concurrency semaphore so a
        retrying call does not hold a slot. Leaving the SDK's own retries on would multiply the
        two policies together -- ``binding.max_retries`` attempts each silently becoming several
        SDK attempts -- so the real worst-case wall-clock per call would be a product nobody
        wrote down. ``timeout=binding.request_timeout`` is the PER-ATTEMPT bound; the caller's
        ``asyncio.wait_for`` enforces the same number, and having both means neither a hung
        socket nor a hung coroutine can outlive it.

        The SDK imports are lazy so the EDA can import this module without openai installed.
    """
    import os

    key = api_key
    if key is None and binding.api_key_env:
        key = os.environ.get(binding.api_key_env)
    if key is None:
        default_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        key = os.environ.get(default_env.get(binding.provider, ""), "") or None
    if key is None and binding.provider == "openai_compat":
        key = "EMPTY"
    if key is None:
        raise RuntimeError(
            f"No API key for role binding {binding.provider}:{binding.model}. Pass api_key=, "
            f"set api_key_env=, or export the provider's default variable."
        )

    import asyncio

    try:
        loop_id: Optional[int] = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None                       # sync context -- probes and setup cells

    base_key = (binding.provider, binding.model, binding.base_url,
                float(binding.request_timeout), key)
    ck = base_key + (loop_id,)
    if ck in _CLIENT_CACHE:
        return _CLIENT_CACHE[ck]

    # Evict this binding's clients from OTHER loops: their pooled connections are unusable here,
    # and a dead loop's client would otherwise live in the cache for the whole process.
    for stale in [k for k in _CLIENT_CACHE if k[:-1] == base_key and k[-1] != loop_id]:
        del _CLIENT_CACHE[stale]

    kwargs = {"api_key": key, "timeout": float(binding.request_timeout), "max_retries": 0}
    if binding.base_url:
        kwargs["base_url"] = binding.base_url

    if binding.provider in ("openai", "openai_compat"):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(**kwargs)
    elif binding.provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(**kwargs)
    else:
        raise ValueError(f"Unknown provider {binding.provider!r}")

    _CLIENT_CACHE[ck] = client
    return client


def reset_client_cache() -> int:
    """Drop every cached client; returns how many were dropped.

    Call this after ``vllm_serve.ensure_alive`` has restarted a server. The base_url is
    unchanged, so the cache would happily hand back the old client and its pool of connections
    to a process that no longer exists -- which surfaces as a burst of connection errors on the
    next phase rather than as anything that names the restart.
    """
    n = len(_CLIENT_CACHE)
    _CLIENT_CACHE.clear()
    return n
