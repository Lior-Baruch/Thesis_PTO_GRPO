"""runtime.py -- host detection, credentials, workspace root, and the import-order guard.

Everything in a run that depends on *where it is running* rather than on *what it is training*.
Four jobs, each of which has bitten this project at least once:

1. **Host detection.** Colab and the local Windows box differ in how secrets arrive, whether Drive
   has to be mounted, and whether the sm_120 import-order landmine is armed. One function answers
   "which am I", and every host-conditional branch in Exp4 goes through it instead of re-deriving
   the answer from ``sys.modules`` at each call site (Exp3 had that check copy-pasted in four
   places, and one copy drifted).

2. **Workspace-root resolution.** Every path in Exp4 is built off one directory -- the one holding
   ``code/``, ``data/`` and ``eda/``. Exp3 identified it by requiring ``HF_key.txt`` AND
   ``openai_key.txt`` to sit side by side, which worked only because every Exp3 role was an OpenAI
   API call. Exp4's default stack calls no vendor API at all, so that marker would resolve to
   nothing on a machine that never had an OpenAI key. The marker here is structural instead of
   credential-based (see :func:`resolve_workspace_root`), so a fresh clone with zero secrets
   resolves correctly from any cwd inside the tree.

3. **Authentication.** Best-effort by design: the point of Exp4 is that a full arm runs with no
   vendor key whatsoever. :func:`authenticate` therefore *returns* what it managed to set up rather
   than raising, and only hard-fails on a credential the caller explicitly declared it needs --
   i.e. when a role really is bound to a vendor API. Hugging Face is still in the picture even on
   the all-open stack, because ``meta-llama/Llama-3.2-1B`` is a gated repo.

   **There is no Weights & Biases here.** Exp3's ``runtime_setup.authenticate`` also logged into
   W&B and set ``WANDB_LOG_MODEL``; Exp4 logs to TensorBoard only, so that whole branch is
   deliberately dropped rather than ported.

4. **The import-order guard.** On the local Blackwell card (sm_120), ``import trl`` *after* torch
   has already been imported segfaults at CUDA init -- exit 139, no Python traceback, nothing to
   catch. It is a native init-order conflict, not an OOM and not a bug in the trainers, and Colab
   is unaffected. :func:`assert_import_order` turns that silent process death into a readable
   exception, and it does so by inspecting ``sys.modules`` only -- it never imports torch itself,
   so it is safe to call from the very top of a notebook.

Running OFF Colab (a GPU server over SSH)
-----------------------------------------
Nothing here is Colab-specific except the secret and mount branches, and both fall through
cleanly on any other Linux/Windows host:

* :func:`detect_host` returns ``"local"`` -- there is no third value. Every "local" branch in
  Exp4 means "not Colab", not "Lior's Windows laptop"; the sm_120 import-order guard is the one
  local-only behaviour, and it is a no-op unless the pair it checks for is actually present
  (bypass with ``EXP4_SKIP_IMPORT_ORDER_CHECK=1`` on a card that does not reproduce it).
* Credentials come from environment variables, which :func:`get_secret` consults FIRST, before
  Colab Secrets and before any key file: ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN`` /
  ``HUGGINGFACE_TOKEN``), ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``. Export them in the SSH
  session (or the service's unit file) and :func:`authenticate` needs nothing else. A key file
  (``HF_key.txt`` beside ``CLAUDE.md``) still works as the fallback it is on the laptop.
* The workspace root resolves by walking up from the cwd (or from this file), so a clone at any
  path works; ``EXP4_WORKSPACE_ROOT=/abs/path/Exp4_OpenStack`` overrides the search outright, for
  a job launched from a scheduler whose cwd is elsewhere.
* ``data/`` is not a Drive symlink there -- point the three subdirectories at real storage before
  the first iteration; nothing in Exp4 requires Drive.
* :func:`describe_environment` records the GPU's total memory in GiB and the installed vLLM
  version into the run banner and ``run_metadata.json``, so an arm trained on an 80 GB A100 is
  distinguishable from one trained on a 40 GB card after the fact -- the VRAM budget and the
  server's ``gpu_memory_utilization`` are functions of that number.

Module-level imports are pure stdlib, and every heavy import (huggingface_hub, torch,
google.colab) happens inside a function. The read-only EDA imports this module for
:func:`resolve_workspace_root` and must not pay for a torch import to get a path.

Usage (trainer notebooks, cell 2 -- before ``serve_roles`` and before any torch import)::

    from core.runtime import detect_host, resolve_workspace_root, authenticate, describe_environment

    ROOT = resolve_workspace_root()
    auth = authenticate(hf=True)                  # openai=/anthropic= only if a role is API-bound
    print(format_environment(describe_environment()))
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "WORKSPACE_DIR_NAME",
    "SECRET_SPECS",
    "SecretSpec",
    "detect_host",
    "mount_drive_if_colab",
    "resolve_workspace_root",
    "get_secret",
    "authenticate",
    "assert_import_order",
    "describe_environment",
    "format_environment",
]


# The directory this whole experiment hangs off. Used as the primary (name-based) root marker and
# quoted in every "could not resolve" error, so it is a constant rather than four string literals.
WORKSPACE_DIR_NAME = "Exp4_OpenStack"

# Escape hatches. Both are read fresh on every call: a notebook that sets one in cell 1 must be
# obeyed by a module imported in cell 2, and caching would defeat that.
HOST_ENV_VAR = "EXP4_HOST"                       # "colab" | "local"
ROOT_ENV_VAR = "EXP4_WORKSPACE_ROOT"             # absolute path, wins over any search
SKIP_IMPORT_ORDER_ENV_VAR = "EXP4_SKIP_IMPORT_ORDER_CHECK"

# Deep enough to reach a filesystem root from anywhere inside the tree (the deepest real start is
# code/tools/fixtures/sanity/, four levels down), shallow enough that a bad `start` cannot turn
# into a long stat storm.
_MAX_WALK_STEPS = 12

# Only the head of CLAUDE.md is read for the fallback marker; the file is ~550 lines and the
# identity line is the first one.
_CLAUDE_MD_HEAD_BYTES = 4096

_TRUTHY = frozenset({"1", "true", "yes", "on"})


# +----------------------------------------------------------------------------+
# |                             HOST DETECTION                                 |
# +----------------------------------------------------------------------------+


@functools.lru_cache(maxsize=1)
def _google_colab_installed() -> bool:
    """Is the ``google.colab`` package importable? Cached -- it is a filesystem probe.

    Notes:
        Uses ``find_spec`` so the package is located but never executed. This is the fallback for
        the case ``sys.modules`` cannot cover: a plain ``python foo.py`` launched from a Colab
        notebook is a fresh interpreter that never imported ``google.colab``, yet is unmistakably
        running on Colab.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("google.colab") is not None
    except Exception:
        # A namespace-package ``google`` from protobuf/googleapis can make find_spec raise rather
        # than return None on some installs. Absence of proof is proof of absence here.
        return False


def detect_host() -> str:
    """Return ``"colab"`` or ``"local"``.

    Returns:
        ``"colab"`` on a Google Colab runtime, ``"local"`` otherwise.

    Raises:
        ValueError: if ``EXP4_HOST`` is set to anything other than ``colab``/``local``. A typo in
            an explicit override is a configuration error, not something to silently ignore --
            silently falling through would send the run down the wrong secret-resolution path.

    Notes:
        Three probes, in order: the ``EXP4_HOST`` override, ``google.colab`` already in
        ``sys.modules`` (true in every Colab *notebook* kernel), then Colab's own environment
        markers plus an importability check (which catches subprocesses and ``!python`` calls).
        Cheap and side-effect free -- it never mounts Drive and never imports torch.
    """
    override = (os.environ.get(HOST_ENV_VAR) or "").strip().lower()
    if override:
        if override not in ("colab", "local"):
            raise ValueError(
                f"{HOST_ENV_VAR} must be 'colab' or 'local', got: {override!r}"
            )
        return override

    if "google.colab" in sys.modules:
        return "colab"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return "colab"
    if _google_colab_installed():
        return "colab"
    return "local"


def mount_drive_if_colab(mount_point: str = "/content/drive") -> Optional[str]:
    """Mount Google Drive when on Colab; no-op everywhere else.

    Args:
        mount_point: Where to mount. The default is the only path the rest of Exp4 assumes.

    Returns:
        The mount point if Drive is mounted (or already was), else ``None``.

    Notes:
        Idempotent -- a second call on an already-mounted runtime does nothing. Kept OUT of
        :func:`detect_host` and :func:`resolve_workspace_root` on purpose: a detector that
        silently blocks on an interactive OAuth prompt is a detector nobody can call from library
        code. The Colab notebook calls this explicitly, before resolving the root, because on
        Colab the workspace lives inside Drive and does not exist until the mount lands.
    """
    if detect_host() != "colab":
        return None
    try:
        from google.colab import drive as _gdrive  # type: ignore

        if not os.path.ismount(mount_point):
            _gdrive.mount(mount_point)
        return mount_point
    except Exception as exc:  # an unmounted Drive is recoverable; a hard failure here is not
        print(f"WARNING: Google Drive mount failed ({exc}).")
        return None


# +----------------------------------------------------------------------------+
# |                           WORKSPACE ROOT                                   |
# +----------------------------------------------------------------------------+


def _walk_up(start: str, max_steps: int = _MAX_WALK_STEPS) -> List[str]:
    """Return ``[start, parent, grandparent, ...]`` as absolute paths, bounded by ``max_steps``."""
    out: List[str] = []
    cur = os.path.abspath(start)
    for _ in range(max_steps):
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:  # filesystem root, or a drive root on Windows
            break
        cur = parent
    return out


def _mentions_exp4(path: str) -> bool:
    """Does the head of the file at ``path`` name this experiment? Never raises."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return WORKSPACE_DIR_NAME in fh.read(_CLAUDE_MD_HEAD_BYTES)
    except OSError:
        return False


def _is_workspace_root(candidate: str) -> bool:
    """Is ``candidate`` the Exp4 workspace root?

    Two independent markers, both requiring a ``code/`` subdirectory so that a bare parent
    directory can never match:

    * the directory is named ``Exp4_OpenStack`` -- true for the repo checkout and for the Colab
      Drive mirror, which is a 1:1 copy of the same folder name; or
    * it holds a ``CLAUDE.md`` whose head names ``Exp4_OpenStack`` -- which keeps resolution
      working if the folder is renamed or unpacked somewhere flat.

    Notes:
        Deliberately NOT credential-based (Exp3 keyed on ``HF_key.txt`` + ``openai_key.txt`` being
        present together). Exp4's default stack has no vendor keys at all, so a credential marker
        would fail on exactly the configuration the experiment exists to demonstrate.

        Deliberately NOT ``data/``-based either: ``data/`` is three Google Drive symlinks that a
        fresh clone has not created yet, and a resolver that fails before the symlinks exist
        cannot be used by the script that would create them.

        ``Exp3_PTO_GRPO/`` matches neither branch (wrong name, and it has no ``CLAUDE.md`` of its
        own), so a cwd inside Exp3 cannot resolve to an Exp4 root by accident.
    """
    if not os.path.isdir(os.path.join(candidate, "code")):
        return False
    if os.path.basename(os.path.normpath(candidate)) == WORKSPACE_DIR_NAME:
        return True
    return _mentions_exp4(os.path.join(candidate, "CLAUDE.md"))


def resolve_workspace_root(start: Optional[str] = None) -> str:
    """Resolve the ``Exp4_OpenStack/`` directory by walking up from ``start``.

    Args:
        start: Directory (or file, whose directory is used) to start the upward walk from.
            Defaults to ``os.getcwd()``.

    Returns:
        Absolute path to the workspace root.

    Raises:
        RuntimeError: if no ancestor of ``start`` and no ancestor of this module's own file
            matches, or if ``EXP4_WORKSPACE_ROOT`` points at something that is not a directory.

    Notes:
        Resolution order:

        1. ``EXP4_WORKSPACE_ROOT`` if set -- an explicit override always wins.
        2. Nearest ancestor of ``start`` (inclusive) satisfying :func:`_is_workspace_root`. This
           is what makes a cwd of ``code/grpo/``, ``code/tools/`` or ``eda/`` all resolve.
        3. Nearest ancestor of **this module's own directory**. ``runtime.py`` physically lives at
           ``<root>/code/core/runtime.py``, so this anchor holds even when cwd is somewhere
           entirely unrelated -- a scheduled job, a debugger launched from the repo root, a test
           runner with its own tmp cwd. Exp3 had no such fallback and printed a warning plus a
           wrong root instead.

        Not cached: the walk is a handful of ``os.path.isdir`` calls, and on Colab the answer
        legitimately changes the moment Drive mounts.
    """
    override = (os.environ.get(ROOT_ENV_VAR) or "").strip()
    if override:
        override = os.path.abspath(os.path.expanduser(override))
        if not os.path.isdir(override):
            raise RuntimeError(
                f"{ROOT_ENV_VAR} is set to {override!r}, which is not a directory."
            )
        if not _is_workspace_root(override):
            print(
                f"WARNING: {ROOT_ENV_VAR}={override!r} does not look like an {WORKSPACE_DIR_NAME} "
                f"root (expected a code/ subdirectory). Honouring it anyway."
            )
        return override

    start_dir = os.path.abspath(start) if start else os.getcwd()
    if os.path.isfile(start_dir):
        start_dir = os.path.dirname(start_dir)

    module_dir = os.path.dirname(os.path.abspath(__file__))
    searched: List[str] = []
    for anchor in (start_dir, module_dir):
        for candidate in _walk_up(anchor):
            searched.append(candidate)
            if _is_workspace_root(candidate):
                return candidate

    raise RuntimeError(
        f"Could not resolve the {WORKSPACE_DIR_NAME} workspace root.\n"
        f"  Started from : {start_dir}\n"
        f"  Module anchor: {module_dir}\n"
        f"  Looked for an ancestor directory containing code/ and either named "
        f"{WORKSPACE_DIR_NAME} or holding a CLAUDE.md that names it.\n"
        f"  Searched {len(searched)} directories.\n"
        f"  Fix: run from inside the experiment tree, pass start=..., or set "
        f"{ROOT_ENV_VAR} to the absolute path."
    )


# +----------------------------------------------------------------------------+
# |                                SECRETS                                     |
# +----------------------------------------------------------------------------+


@dataclass(frozen=True)
class SecretSpec:
    """Where one logical credential can be found on either host.

    Attributes:
        name: Canonical logical name (``"hf"``, ``"openai"``, ``"anthropic"``).
        env_vars: Environment variables to try, in order.
        colab_keys: Colab Secrets (``google.colab.userdata``) names to try, in order.
        filenames: Bare filenames to look for under the workspace root and its parent.
    """

    name: str
    env_vars: Tuple[str, ...]
    colab_keys: Tuple[str, ...]
    filenames: Tuple[str, ...]


#: The three credentials Exp4 can use. Only ``hf`` matters on the default all-open stack, and only
#: because ``meta-llama/Llama-3.2-1B`` is a gated repo -- the oracle, patient and judge all run on
#: a local vLLM server that authenticates nothing.
#:
#: Filenames match Exp3's convention exactly so a key copied from ``Exp3_PTO_GRPO/`` keeps working
#: (they are gitignored repo-wide by the ``**/HF_key.txt`` rules).
SECRET_SPECS: Dict[str, SecretSpec] = {
    "hf": SecretSpec(
        name="hf",
        env_vars=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"),
        colab_keys=("huggingface", "HF_TOKEN"),
        filenames=("HF_key.txt",),
    ),
    "openai": SecretSpec(
        name="openai",
        env_vars=("OPENAI_API_KEY",),
        colab_keys=("OPENAI_API_KEY", "openai"),
        filenames=("openai_key.txt",),
    ),
    "anthropic": SecretSpec(
        name="anthropic",
        env_vars=("ANTHROPIC_API_KEY",),
        colab_keys=("ANTHROPIC_API_KEY", "anthropic"),
        filenames=("anthropic_key.txt",),
    ),
}


def _build_secret_aliases() -> Dict[str, str]:
    """Every spelling that maps onto a canonical secret name, lowercased."""
    aliases: Dict[str, str] = {}
    for canonical, spec in SECRET_SPECS.items():
        names = [canonical, *spec.env_vars, *spec.colab_keys, *spec.filenames]
        for raw in names:
            aliases[raw.lower()] = canonical
            if raw.lower().endswith(".txt"):
                aliases[raw.lower()[: -len(".txt")]] = canonical
    aliases["huggingface_hub"] = "hf"
    aliases["hf_key"] = "hf"
    return aliases


_SECRET_ALIASES: Dict[str, str] = _build_secret_aliases()


def _spec_for(name: str) -> SecretSpec:
    """Resolve ``name`` to a :class:`SecretSpec`, inventing an env-only one for unknown names."""
    canonical = _SECRET_ALIASES.get(name.strip().lower())
    if canonical is not None:
        return SECRET_SPECS[canonical]
    # An unregistered name is treated as a bare environment variable. That keeps get_secret usable
    # for one-off knobs without forcing every caller to register a spec, and it cannot accidentally
    # read a file (no filenames), which is the part with a footgun.
    return SecretSpec(name=name, env_vars=(name,), colab_keys=(), filenames=())


def _colab_userdata(key: str) -> str:
    """Read one Colab Secret. Returns ``""`` on any failure (not on Colab, not granted, absent)."""
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(key)
    except Exception:
        return ""
    return str(value).strip() if value else ""


def _read_secret_file(path: str) -> str:
    """First non-empty line of ``path``, stripped. ``""`` if unreadable.

    Notes:
        First *line*, not the whole file: a key pasted from a browser routinely carries a trailing
        newline, and an operator sometimes leaves a comment under it. Taking the whole file would
        send those bytes to the provider and produce a 401 that looks like a bad key.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    return stripped
    except OSError:
        return ""
    return ""


def _secret_file_candidates(spec: SecretSpec) -> List[str]:
    """Absolute paths to try for ``spec``'s key files, nearest-first."""
    if not spec.filenames:
        return []
    dirs: List[str] = []
    try:
        root = resolve_workspace_root()
        dirs.append(root)
        dirs.append(os.path.dirname(root))  # the repo root, one level up
    except RuntimeError:
        pass
    dirs.append(os.getcwd())

    seen: set = set()
    out: List[str] = []
    for directory in dirs:
        key = os.path.normcase(os.path.abspath(directory))
        if key in seen:
            continue
        seen.add(key)
        for filename in spec.filenames:
            out.append(os.path.join(directory, filename))
    return out


def get_secret(name: str, *, required: bool = False) -> Optional[str]:
    """Resolve one credential by logical name.

    Args:
        name: ``"hf"``, ``"openai"``, ``"anthropic"``, or any of their env-var / Colab-secret /
            filename spellings (case-insensitive -- ``"HF_TOKEN"``, ``"openai_key.txt"`` and
            ``"anthropic"`` all work). An unregistered name is read from the environment only.
        required: Raise instead of returning ``None`` when nothing resolves.

    Returns:
        The credential string, or ``None`` when it is absent and ``required`` is false.

    Raises:
        RuntimeError: when ``required`` is true and nothing resolved. The message lists every
            place that was searched -- never the value of anything found.

    Notes:
        Search order is environment variable, then Colab Secrets, then a key file under the
        workspace root (and its parent, and the cwd). Environment first so a shell export or a
        prior :func:`authenticate` call is authoritative, and so a CI run can override a stale
        checked-out key file without deleting it.

        Colab Secrets are consulted only when :func:`detect_host` says ``"colab"``; off Colab the
        import would fail on every call for no benefit.
    """
    spec = _spec_for(name)

    for var in spec.env_vars:
        value = (os.environ.get(var) or "").strip()
        if value:
            return value

    if detect_host() == "colab":
        for key in spec.colab_keys:
            value = _colab_userdata(key)
            if value:
                return value

    for path in _secret_file_candidates(spec):
        value = _read_secret_file(path)
        if value:
            return value

    if required:
        raise RuntimeError(
            f"Required credential {spec.name!r} not found.\n"
            f"  Environment variables tried: {', '.join(spec.env_vars) or '(none)'}\n"
            f"  Colab Secrets tried        : {', '.join(spec.colab_keys) or '(none)'}\n"
            f"  Key files tried            : "
            f"{', '.join(_secret_file_candidates(spec)) or '(none)'}\n"
            f"  Note: the default Exp4 stack needs NO vendor key. If you did not intend to bind "
            f"a role to a vendor API, check the role bindings in cell 1."
        )
    return None


# +----------------------------------------------------------------------------+
# |                            AUTHENTICATION                                  |
# +----------------------------------------------------------------------------+


def authenticate(
    *, hf: bool = True, openai: bool = False, anthropic: bool = False
) -> Dict[str, bool]:
    """Put the requested credentials in place for this process and its children.

    Args:
        hf: Resolve the Hugging Face token, export ``HF_TOKEN``, and attempt a hub login.
            **Best effort** -- a missing token warns and returns ``False``.
        openai: Export ``OPENAI_API_KEY``. **Required** when true: a missing key raises.
        anthropic: Export ``ANTHROPIC_API_KEY``. **Required** when true: a missing key raises.

    Returns:
        ``{"hf": bool, "openai": bool, "anthropic": bool}``. ``True`` means the credential is now
        in this process's environment; ``False`` means either it was not requested or it did not
        resolve. Check the flag you actually passed.

    Raises:
        RuntimeError: when ``openai`` or ``anthropic`` is requested and no key resolves.

    Notes:
        The asymmetry is the whole point of Exp4. ``openai=True`` / ``anthropic=True`` are stated
        only when a :class:`~roles.RoleBinding` really points at that vendor, so a missing key
        there is a run-stopping misconfiguration. ``hf=True`` is the default because
        ``meta-llama/Llama-3.2-1B`` is gated -- but the weights may already be in the local HF
        cache, and the EDA needs no token at all, so a missing HF token warns rather than killing
        an otherwise valid session. A caller that genuinely cannot proceed without it should say
        so explicitly::

            get_secret("hf", required=True)

        ``HF_TOKEN`` is exported into ``os.environ`` (not just handed to ``login``) because
        ``tools/vllm_serve.py`` starts the vLLM server as a **subprocess**, which inherits the
        environment but not the in-process hub session.

        There is no Weights & Biases branch here: Exp4 logs to TensorBoard only.
    """
    result: Dict[str, bool] = {"hf": False, "openai": False, "anthropic": False}

    if hf:
        token = get_secret("hf")
        if token:
            os.environ["HF_TOKEN"] = token
            result["hf"] = True
            try:
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)
                print("Hugging Face: logged in (HF_TOKEN exported).")
            except Exception as exc:
                # HF_TOKEN in the environment is already enough for from_pretrained and for the
                # vLLM subprocess; the cached login is a convenience on top of it.
                print(
                    f"WARNING: huggingface_hub login failed ({exc}). "
                    f"HF_TOKEN is exported, so gated downloads should still work."
                )
        else:
            spec = SECRET_SPECS["hf"]
            print(
                f"WARNING: no Hugging Face token found. Looked at the environment variables "
                f"{', '.join(spec.env_vars)}, then Colab Secrets "
                f"({', '.join(spec.colab_keys)}; Colab only), then {', '.join(spec.filenames)} "
                f"under the workspace root. meta-llama/Llama-3.2-1B is gated -- this only works "
                f"if the weights are already in the local HF cache. Off Colab, export "
                f"{spec.env_vars[0]} in the shell."
            )

    if openai:
        os.environ["OPENAI_API_KEY"] = get_secret("openai", required=True) or ""
        result["openai"] = True
        print("OpenAI: OPENAI_API_KEY exported.")

    if anthropic:
        os.environ["ANTHROPIC_API_KEY"] = get_secret("anthropic", required=True) or ""
        result["anthropic"] = True
        print("Anthropic: ANTHROPIC_API_KEY exported.")

    return result


# +----------------------------------------------------------------------------+
# |                         IMPORT-ORDER GUARD                                 |
# +----------------------------------------------------------------------------+


_IMPORT_ORDER_MESSAGE = (
    "Import order violation: torch is already imported but trl is not.\n"
    "\n"
    "On the local Blackwell card (sm_120), importing trl AFTER torch segfaults at CUDA init -- "
    "exit 139, no Python traceback, no OOM, nothing to catch. It is a native initialisation-order "
    "conflict, not a bug in the trainers, and Colab is unaffected (which is why the full runs ran "
    "there).\n"
    "\n"
    "Fix: import trl FIRST, before torch and before anything that pulls torch in "
    "(transformers, peft, accelerate, the core.policy / core.lookahead modules).\n"
    "\n"
    "    import trl          # must come first on a local sm_120 host\n"
    "    import torch\n"
    "\n"
    "In a notebook, restart the kernel and re-run: an already-imported torch cannot be unwound.\n"
    "The notebook cell order that satisfies this is cell 1 globals -> runtime detect/auth -> "
    "serve_roles() -> import trl -> torch/model build -> the orchestration loop.\n"
    "\n"
    f"Set {SKIP_IMPORT_ORDER_ENV_VAR}=1 to bypass this check on a non-Blackwell local GPU."
)

_PYARROW_ORDER_MESSAGE = (
    "Import order violation: torch is already imported but datasets/pyarrow is not.\n"
    "\n"
    "MEASURED on the local RTX 5070 Ti (sm_120): `import torch, datasets` and "
    "`import trl, torch, datasets` both die with a Windows access violation (exit 139) inside "
    "pyarrow.dataset, while `import datasets, torch` and `import trl, datasets, torch` are fine. "
    "pyarrow and torch each initialise native runtimes, and the survivor is whichever goes first "
    "-- the same class of conflict as the trl one above, a different pair of libraries.\n"
    "\n"
    "This is easy to miss because the TRAINERS do not hit it: they pull pandas (and therefore "
    "pyarrow) in through core.* before their own `import torch`. That is an accident of import "
    "order, not a design, and it holds only until someone reorders those lines. tools/smoke.py "
    "had no such accident and segfaulted before reaching its first check.\n"
    "\n"
    "Fix: import datasets BEFORE torch. The full safe order is:\n"
    "\n"
    "    import trl          # first  -- native CUDA init\n"
    "    import datasets     # second -- native pyarrow init\n"
    "    import torch        # third\n"
    "\n"
    "In a notebook, restart the kernel and re-run: an already-imported torch cannot be unwound.\n"
    "\n"
    f"Set {SKIP_IMPORT_ORDER_ENV_VAR}=1 to bypass this check on a local GPU that does not "
    "reproduce it."
)


def _module_installed(name: str) -> bool:
    """Is *name* importable, WITHOUT importing it.

    ``find_spec`` reads metadata only, so this cannot itself create the import-order condition the
    caller is about to assert on.
    """
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def assert_import_order() -> None:
    """Fail loudly on the local import order that segfaults at CUDA init.

    Raises:
        RuntimeError: on a local host when ``torch`` is in ``sys.modules`` and ``trl`` is not.

    Notes:
        Inspects ``sys.modules`` only -- it never imports torch, trl, or anything else, so it is
        cheap enough to call at the top of every notebook and every tool script, and calling it
        cannot itself create the condition it checks for.

        No-op on Colab (the conflict is specific to the local sm_120 driver stack) and bypassable
        via ``EXP4_SKIP_IMPORT_ORDER_CHECK=1`` for a local machine with a different card.

        The check is one-directional on purpose: ``trl`` imported without ``torch`` is impossible
        (trl imports torch), and neither imported yet is the healthy starting state.
    """
    if detect_host() == "colab":
        return
    if (os.environ.get(SKIP_IMPORT_ORDER_ENV_VAR) or "").strip().lower() in _TRUTHY:
        return
    if "torch" in sys.modules and "trl" not in sys.modules:
        raise RuntimeError(_IMPORT_ORDER_MESSAGE)
    # Second native-init pair, same failure signature (exit 139), different libraries. Asserted
    # only when datasets is actually installed: a host without it cannot trip the conflict, and
    # find_spec answers that without importing anything.
    if ("torch" in sys.modules and "datasets" not in sys.modules
            and _module_installed("datasets")):
        raise RuntimeError(_PYARROW_ORDER_MESSAGE)


# +----------------------------------------------------------------------------+
# |                         ENVIRONMENT REPORT                                 |
# +----------------------------------------------------------------------------+


def _probe_gpus() -> Optional[List[Dict[str, Any]]]:
    """GPU name + total VRAM per device via ``nvidia-smi``, or ``None`` if it cannot be asked.

    Notes:
        Shells out rather than touching torch **on purpose**. ``torch.cuda.is_available()`` and
        friends can initialise the CUDA driver in this process, and on the local card an
        accidental early CUDA context is exactly the class of thing that costs a reboot rather
        than an exception. ``nvidia-smi`` answers the question from outside the process and
        leaves this interpreter's CUDA state untouched.

        Returns ``None`` (unknown) rather than ``[]`` when ``nvidia-smi`` is missing or fails --
        "no driver installed" and "driver present, zero GPUs" are different facts.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    gpus: List[Dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # rsplit: some marketing names contain commas; the memory field never does.
        name, _, mem = line.rpartition(",")
        if not name:
            continue
        try:
            total_mib: Optional[int] = int(float(mem.strip()))
        except ValueError:
            total_mib = None
        gpus.append({"name": name.strip(), "total_mib": total_mib})
    return gpus


#: Packages whose installed version the banner records. Read through ``importlib.metadata`` --
#: never imported -- so asking costs nothing and cannot change this process's import order.
#: ``vllm`` is the one that matters most: it is the grader's server, it is not in
#: ``requirements.txt``'s pinned set on every host, and its guided-decoding behaviour (the
#: ``strict`` key, ``minItems`` enforcement) is version-dependent.
_VERSIONED_PACKAGES = ("vllm", "torch", "transformers", "trl", "peft", "openai")


def _installed_version(package: str) -> Optional[str]:
    """Installed distribution version of *package*, or ``None`` -- WITHOUT importing it."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # PackageNotFoundError, or a broken metadata dir; both mean "unknown"
        return None


def _torch_gpu_total_gib() -> Optional[float]:
    """Device 0's total memory in GiB as torch reports it -- only if torch is ALREADY imported.

    Read off ``sys.modules`` and never imported here: importing torch from a path helper would
    arm the sm_120 import-order landmine for every later ``import trl``. When torch is present,
    ``get_device_properties`` is the authoritative figure (it is the number the allocator budgets
    against), and it is what tells an 80 GB A100 from a 40 GB one in ``run_metadata.json``.
    ``None`` when torch is absent, CUDA is unavailable, or the query fails.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.get_device_properties(0).total_memory) / float(1024 ** 3)
    except Exception:
        return None


def describe_environment() -> Dict[str, Any]:
    """Collect everything the run banner should state about this machine.

    Returns:
        A JSON-serialisable dict:

        * ``host`` -- ``"colab"`` | ``"local"``
        * ``python`` / ``python_executable`` / ``platform``
        * ``workspace_root`` -- absolute path, or ``None`` if unresolvable
        * ``workspace_root_error`` -- the resolver's message when it is ``None``
        * ``cuda_visible`` -- ``True`` / ``False`` / ``None`` (unknown: no ``nvidia-smi``)
        * ``cuda_visible_devices`` -- the raw env var, or ``None`` if unset
        * ``gpus`` -- ``[{"name": str, "total_mib": int|None}, ...]`` or ``None``
        * ``gpu_name`` / ``gpu_total_mib`` -- device 0, hoisted for the banner
        * ``gpu_total_gib`` -- device 0's total memory in GiB (``None`` if unknown), and
          ``gpu_total_gib_source`` -- ``"torch"`` when torch was already imported and CUDA
          answered (the allocator's own figure), else ``"nvidia-smi"``, else ``None``. This is
          the field that tells an 80 GB A100 from a 40 GB one.
        * ``vllm_version`` -- the installed vLLM distribution version, or ``None``; plus
          ``package_versions`` -- ``{name: version|None}`` for :data:`_VERSIONED_PACKAGES`.
          Read via ``importlib.metadata``, never by importing.
        * ``torch_imported`` / ``trl_imported`` -- ``sys.modules`` membership, for the guard
        * ``secrets`` -- ``{"hf": bool, "openai": bool, "anthropic": bool}``

    Notes:
        ``secrets`` reports **only whether each credential resolved**. No value, no prefix, no
        length -- this dict is printed in notebooks, pasted into issues and archived next to
        ``run_metadata.json``, and a masked key is still a leaked key once someone knows the
        vendor's prefix length.

        Nothing here imports torch (or vllm). ``torch_imported`` is a ``sys.modules`` lookup, so
        a ``False`` is meaningful: it says the process is still in the state where
        :func:`assert_import_order` can be satisfied. The GiB figure is read from torch ONLY when
        the caller already imported it (``sys.modules``), and from ``nvidia-smi`` otherwise --
        see :func:`_probe_gpus` for why the latter deliberately stays out of process.

        Cheap but not free -- ``nvidia-smi`` is a subprocess. Call it once for the banner, not per
        iteration.
    """
    try:
        workspace_root: Optional[str] = resolve_workspace_root()
        root_error: Optional[str] = None
    except RuntimeError as exc:
        workspace_root = None
        root_error = str(exc)

    gpus = _probe_gpus()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        cuda_visible: Optional[bool] = False  # devices exist but are masked off for this process
    elif gpus is None:
        cuda_visible = None
    else:
        cuda_visible = bool(gpus)

    gpu_total_mib = gpus[0]["total_mib"] if gpus else None
    gpu_total_gib = _torch_gpu_total_gib()
    gib_source: Optional[str] = "torch" if gpu_total_gib is not None else None
    if gpu_total_gib is None and gpu_total_mib:
        gpu_total_gib = float(gpu_total_mib) / 1024.0
        gib_source = "nvidia-smi"

    versions = {name: _installed_version(name) for name in _VERSIONED_PACKAGES}

    import platform  # stdlib, but only needed here

    info: Dict[str, Any] = {
        "host": detect_host(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "workspace_root": workspace_root,
        "workspace_root_error": root_error,
        "cuda_visible": cuda_visible,
        "cuda_visible_devices": cvd,
        "gpus": gpus,
        "gpu_name": gpus[0]["name"] if gpus else None,
        "gpu_total_mib": gpu_total_mib,
        "gpu_total_gib": (None if gpu_total_gib is None else round(gpu_total_gib, 2)),
        "gpu_total_gib_source": gib_source,
        "vllm_version": versions["vllm"],
        "package_versions": versions,
        "torch_imported": "torch" in sys.modules,
        "trl_imported": "trl" in sys.modules,
        "secrets": {name: get_secret(name) is not None for name in SECRET_SPECS},
    }
    return info


def format_environment(info: Optional[Dict[str, Any]] = None) -> str:
    """Render :func:`describe_environment` as a fixed-width ASCII banner.

    Args:
        info: A dict from :func:`describe_environment`; collected fresh when omitted.

    Returns:
        A multi-line string, ASCII only (the Windows console cannot be trusted with anything
        else), safe to print or to write into a log file.
    """
    if info is None:
        info = describe_environment()

    gpu = "unknown"
    if info.get("gpus") is not None:
        if info["gpus"]:
            gpu = ", ".join(
                f"{g['name']} ({g['total_mib']} MiB)" if g["total_mib"] else str(g["name"])
                for g in info["gpus"]
            )
        else:
            gpu = "none visible"

    secrets = info.get("secrets") or {}
    secret_line = ", ".join(
        f"{name}={'yes' if ok else 'no'}" for name, ok in sorted(secrets.items())
    )

    gib = info.get("gpu_total_gib")
    gib_line = (
        f"{gib:.1f} GiB (via {info.get('gpu_total_gib_source')})" if gib is not None else "unknown"
    )
    versions = info.get("package_versions") or {}
    version_line = ", ".join(
        f"{name}={ver or 'absent'}" for name, ver in versions.items()
    )

    rows = [
        ("host", info.get("host")),
        ("python", f"{info.get('python')}  ({info.get('python_executable')})"),
        ("platform", info.get("platform")),
        ("workspace root", info.get("workspace_root") or "UNRESOLVED"),
        ("gpu", gpu),
        ("gpu 0 total memory", gib_line),
        ("CUDA_VISIBLE_DEVICES", info.get("cuda_visible_devices") if info.get("cuda_visible_devices") is not None else "(unset)"),
        ("packages", version_line or "(none probed)"),
        ("torch / trl imported", f"{info.get('torch_imported')} / {info.get('trl_imported')}"),
        ("secrets resolved", secret_line or "(none)"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["=" * 78, "Exp4_OpenStack runtime", "-" * 78]
    lines += [f"{label.ljust(width)} : {value}" for label, value in rows]
    lines.append("=" * 78)
    return "\n".join(lines)
