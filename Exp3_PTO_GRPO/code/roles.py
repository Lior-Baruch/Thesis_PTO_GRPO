"""roles.py — which model plays each LLM role (patient / oracle / judge).

Exp3 runs three LLM roles besides the therapist policy:

- **patient** — simulates the client the therapist talks to. Defines the TASK; swapping it
  changes the environment, not just the measurement, so nothing is comparable across a change.
- **oracle** — grades the training reward (Q1+Q2). Swapping it changes what the policy
  optimizes, so arms trained under different oracles are not comparable either.
- **judge** — grades the *eval* scores after the fact. Already pluggable on the EDA side
  (:class:`eda_analysis.scoring.judge.JudgeSpec`); a judge swap is safe and re-runnable,
  which is why the score lake partitions on ``judge=<tag>`` rather than encoding it here.

This module is the single source of truth for (a) how a role is bound to a provider+model
and (b) how a non-default binding is spelled in run/arm names. It is **stdlib-only and
import-light** on purpose: both the trainer (``code/_shared/``) and the read-only EDA
(``eda/eda_analysis/``) import it, and the EDA must not pull in torch. Provider SDKs are
imported lazily inside :func:`make_client`.

Naming contract (the reason this module exists)
-----------------------------------------------
Adding a role dimension to the experiment is only safe if the *identity* of an arm widens
with it. Otherwise a Gemma-oracle run and a gpt-4o-mini run of the same method+K write to
the same ``eval_scores/.../<Model>/`` folder, and ``Run_Eval``'s skip-existing resume
reports "already scored" against the other model's CSVs — silently.

So every name that identifies an arm carries a suffix from :func:`binding_suffix`, and that
function returns ``""`` when every role is on its default. Default-bound runs therefore keep
byte-identical names, and the ~50k CSVs already in the score lake stay valid. Do not change
:data:`DEFAULT_ORACLE_MODEL` / :data:`DEFAULT_PATIENT_MODEL` without migrating those files.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "DEFAULT_ORACLE_MODEL", "DEFAULT_PATIENT_MODEL",
    "RoleBinding", "ORACLE_DEFAULT", "PATIENT_DEFAULT",
    "model_tag", "binding_suffix", "suffix_from_tags", "assert_name_matches_roles",
    "make_client",
]

# The models every existing Exp3 run used. These two strings are load-bearing: they are what
# `binding_suffix` compares against to decide a run is "default" and needs no name suffix.
DEFAULT_ORACLE_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_PATIENT_MODEL = "gpt-4o-mini-2024-07-18"

# Curated short tags. Anything not listed falls back to `_slugify`, which is stable but
# uglier; add an entry here when a model becomes a regular arm so names stay readable.
_MODEL_TAGS = {
    "gpt-4o-mini-2024-07-18": "gpt4m",
    "gpt-4o-mini": "gpt4m",
    "gpt-4o": "gpt4o",
    "claude-haiku-4-5": "haiku45",
}


def _slugify(model_id: str) -> str:
    """Vendor-stripped alphanumeric tag: ``google/gemma-3n-E4B-it`` -> ``gemma3nE4B``.

    Restricted to ``[A-Za-z0-9]`` so a tag can sit inside an ``EXPERIMENT_NAME`` without
    colliding with the ``_``-delimited field structure the arm-name regex depends on.
    """
    s = model_id.split("/")[-1]
    for suffix in ("-it", "-instruct", "-Instruct", "-hf"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return "".join(ch for ch in s if ch.isalnum()) or "model"


def model_tag(model_id: str) -> str:
    """Short filesystem/regex-safe tag for a model id."""
    if not model_id:
        return "none"
    return _MODEL_TAGS.get(model_id) or _slugify(model_id)


@dataclass(frozen=True)
class RoleBinding:
    """Provider + model for one LLM role.

    ``provider``:
      - ``"openai"``       — the OpenAI API.
      - ``"openai_compat"``— any OpenAI-compatible server (vLLM, llama.cpp, TGI). Same call
        shape, including ``response_format={"type": "json_schema"}`` via guided decoding,
        so the oracle's retry/validate stack works unchanged. Needs ``base_url``.
      - ``"anthropic"``    — the Anthropic API (judge side today; see ``scoring/judge.py``
        for the schema constraints Claude rejects).

    ``api_key_env`` overrides which environment variable supplies the key. Local servers
    usually need no key at all — ``make_client`` sends a placeholder.
    """
    model: str
    provider: str = "openai"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: Optional[float] = None

    @property
    def tag(self) -> str:
        return model_tag(self.model)

    @property
    def is_local(self) -> bool:
        return self.provider == "openai_compat"


ORACLE_DEFAULT = RoleBinding(model=DEFAULT_ORACLE_MODEL)
PATIENT_DEFAULT = RoleBinding(model=DEFAULT_PATIENT_MODEL)


def binding_suffix(oracle_model: Optional[str] = None,
                   patient_model: Optional[str] = None) -> str:
    """Name suffix for non-default role bindings — ``""`` when everything is default.

    Takes model-id STRINGS rather than :class:`RoleBinding`s because the EDA reconstructs
    arms from ``run_metadata.json``, which stores ``oracle_model_id`` / ``patient_model_id``
    as plain strings and knows nothing about bindings.

    >>> binding_suffix("gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18")
    ''
    >>> binding_suffix("google/gemma-3n-E4B-it")
    '_Ogemma3nE4B'
    """
    return suffix_from_tags(
        model_tag(oracle_model) if oracle_model and oracle_model != DEFAULT_ORACLE_MODEL else None,
        model_tag(patient_model) if patient_model and patient_model != DEFAULT_PATIENT_MODEL else None,
    )


def suffix_from_tags(oracle_tag: Optional[str] = None,
                     patient_tag: Optional[str] = None) -> str:
    """The reader-side inverse of :func:`binding_suffix`, from already-short tags.

    The EDA recovers an arm's bindings by parsing the folder NAME (authoritative — it is
    where the data physically lives) rather than from ``run_metadata.json``, so it holds
    tags, not model ids. Keeping both directions in one module is what guarantees the
    round-trip: a name built by ``binding_suffix`` re-derives the identical suffix here.
    """
    parts = []
    if oracle_tag:
        parts.append("O" + oracle_tag)
    if patient_tag:
        # "Pat", not "P": PTO names already end in ``_PT{greedy|indep}``, so a bare ``_P``
        # prefix is ambiguous with the mode token (``_PTgreedy`` parses as patient="Tgreedy").
        parts.append("Pat" + patient_tag)
    return ("_" + "_".join(parts)) if parts else ""


def assert_name_matches_roles(experiment_name: str,
                              oracle_model: Optional[str] = None,
                              patient_model: Optional[str] = None) -> None:
    """Raise unless *experiment_name* carries the suffix its role models require.

    The failure this prevents: changing ``ORACLE_MODEL_ID`` in a trainer notebook while
    leaving ``EXPERIMENT_NAME`` alone. The run would then write conversations and scores
    under the default-grader arm's identity, mixing two differently-rewarded policies in one
    folder — and ``Run_Eval``'s skip-existing resume would call them "already scored".

    Only the dangerous direction is checked. A name carrying a suffix while the models are
    default merely creates an extra, correctly-isolated folder; that is wasteful, not wrong.
    """
    expected = binding_suffix(oracle_model, patient_model)
    if expected and not experiment_name.endswith(expected):
        raise ValueError(
            f"EXPERIMENT_NAME {experiment_name!r} uses non-default role models "
            f"(oracle={oracle_model!r}, patient={patient_model!r}) but does not end with "
            f"{expected!r}. Append it, or this run will collide with the default arm in "
            f"data/eval_scores/. Suffix comes from roles.binding_suffix()."
        )


# Clients are cached per binding: the oracle path builds one per scoring call otherwise, and
# an AsyncOpenAI holds a connection pool worth reusing across ~10k calls per iteration.
_CLIENT_CACHE: dict = {}


def make_client(binding: RoleBinding, *, api_key: Optional[str] = None):
    """Async client for *binding*. Cached per (provider, model, base_url, key).

    ``api_key`` wins over ``binding.api_key_env`` over the provider's default env var. An
    ``openai_compat`` server that wants no auth gets the ``"EMPTY"`` placeholder the OpenAI
    SDK requires (it refuses to construct without some key).
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

    ck = (binding.provider, binding.model, binding.base_url, key)
    if ck in _CLIENT_CACHE:
        return _CLIENT_CACHE[ck]

    if binding.provider in ("openai", "openai_compat"):
        from openai import AsyncOpenAI
        kwargs = {"api_key": key}
        if binding.base_url:
            kwargs["base_url"] = binding.base_url
        client = AsyncOpenAI(**kwargs)
    elif binding.provider == "anthropic":
        from anthropic import AsyncAnthropic
        kwargs = {"api_key": key}
        if binding.base_url:
            kwargs["base_url"] = binding.base_url
        client = AsyncAnthropic(**kwargs)
    else:
        raise ValueError(f"Unknown provider {binding.provider!r}")

    _CLIENT_CACHE[ck] = client
    return client
