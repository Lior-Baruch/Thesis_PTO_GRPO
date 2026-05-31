"""
runtime_setup.py — Runtime detection, path resolution, auth, helper preflight.

Pulled out of the training notebook so the notebook stays orchestration-only.

Usage:
    from runtime_setup import detect_runtime, init_openai_client, authenticate, verify_helpers

    rt = detect_runtime(run_env="auto")
    client = init_openai_client(rt)
    authenticate(rt, hf=True, wandb_enabled=("wandb" in REPORT_TO))
    verify_helpers(["questionnaires", "model_helpers", "generation"], [LOCAL_OUTDIR, CONV_OUTDIR])
"""

import os
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List


KEY_FILES = ("HF_key.txt", "openai_key.txt")


@dataclass(frozen=True)
class RuntimeInfo:
    """Resolved runtime context. Returned by :func:`detect_runtime`.

    ``experiment_root`` is the Exp folder itself — keys (``HF_key.txt`` /
    ``openai_key.txt``) live there, so the resolver walks up until it lands
    on it. ``project_root`` is kept as an alias of ``experiment_root`` so
    older callers keep working.
    """
    in_colab: bool
    project_root: str
    experiment_root: str
    experiment_code: str
    grpo_v2_code: str


def _walk_up(start: str, max_steps: int = 8) -> List[str]:
    """Return ``[start, parent_of_start, ...]`` up to ``max_steps`` levels."""
    out: List[str] = []
    cur = os.path.abspath(start)
    for _ in range(max_steps):
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def _candidate_roots(in_colab: bool, experiment_name: str) -> List[str]:
    """Candidate experiment-root dirs (the dir holding the API-key files).

    Colab: the synced Drive folder, then CWD as a fallback for ad-hoc setups.
    Local: walk up from CWD until we find the keys.
    """
    if in_colab:
        return [
            f"/content/drive/MyDrive/Thesis_PTO_GRPO/{experiment_name}",
            os.getcwd(),
        ]
    return _walk_up(os.getcwd())


def _mount_drive_if_colab() -> None:
    """No-op when not in Colab; otherwise mount Drive (idempotent)."""
    if "google.colab" not in sys.modules:
        return
    try:
        from google.colab import drive as _gdrive  # type: ignore
        if not os.path.ismount("/content/drive"):
            _gdrive.mount("/content/drive")
        else:
            print("Drive already mounted at /content/drive")
    except Exception as e:
        print(f"WARNING: Drive mount failed ({e}). Falling back to /content.")


def detect_runtime(
    run_env: str = "auto",
    experiment_name: str = "Exp3_PTO_GRPO",
) -> RuntimeInfo:
    """Detect Colab vs local, locate the experiment root, prepend code dirs to ``sys.path``.

    The experiment root is identified by two API-key files at its top level
    (``HF_key.txt``, ``openai_key.txt``). On Colab the layout is a 1:1 mirror
    of the local folder, so the same relative paths work everywhere — no path
    translation downstream.

    Args:
        run_env: ``"auto"``, ``"colab"``, or ``"local"``.
        experiment_name: Exp folder name (used only for Colab candidate paths).
    """
    if run_env not in {"auto", "colab", "local"}:
        raise ValueError(f"run_env must be one of auto|colab|local, got: {run_env}")

    detected_colab = "google.colab" in sys.modules
    if run_env == "auto":
        in_colab = detected_colab
    elif run_env == "colab":
        in_colab = True
    else:
        in_colab = False

    print(f"Detected Colab runtime: {detected_colab}")
    print(f"Configured run_env:     {run_env}")
    print(f"Effective in_colab:     {in_colab}")

    if in_colab:
        _mount_drive_if_colab()

    experiment_root = None
    candidates = _candidate_roots(in_colab, experiment_name)
    for root in candidates:
        if all(os.path.exists(os.path.join(root, kf)) for kf in KEY_FILES):
            experiment_root = root
            break

    if experiment_root is None:
        experiment_root = candidates[0]
        print(f"WARNING: Could not find {'/'.join(KEY_FILES)} in candidate roots.")
        print(f"         Using fallback experiment_root: {experiment_root}")
    else:
        print(f"Resolved experiment_root: {experiment_root}")

    experiment_code = os.path.join(experiment_root, "code")
    grpo_v2_code = os.path.join(experiment_code, "GRPO_Exp3")

    # Make per-experiment helpers importable regardless of CWD.
    for p in (grpo_v2_code, experiment_code):
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"sys.path[0..1]:  {sys.path[0]!r}, {sys.path[1]!r}")

    return RuntimeInfo(
        in_colab=in_colab,
        project_root=experiment_root,  # alias kept for backward compat
        experiment_root=experiment_root,
        experiment_code=experiment_code,
        grpo_v2_code=grpo_v2_code,
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            API KEYS / AUTH                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _load_openai_key(rt: RuntimeInfo) -> str:
    """Resolve OPENAI_API_KEY: Colab userdata > env > openai_key.txt."""
    key = ""
    if rt.in_colab:
        try:
            from google.colab import userdata  # type: ignore
            k = userdata.get("OPENAI_API_KEY")
            if k is not None:
                key = str(k).strip()
        except Exception:
            pass

    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not key:
        for kp in (
            os.path.join(rt.project_root, "openai_key.txt"),
            os.path.join(os.getcwd(), "openai_key.txt"),
        ):
            if os.path.exists(kp):
                key = Path(kp).read_text(encoding="utf-8").strip()
                print(f"Loaded OpenAI key from file: {kp}")
                break

    if not key:
        raise RuntimeError(
            "OpenAI API key not found. Provide OPENAI_API_KEY in Colab userdata, "
            "the OPENAI_API_KEY environment variable, or openai_key.txt at the project root."
        )
    return key


def init_openai_client(rt: RuntimeInfo):
    """Return an :class:`AsyncOpenAI` client configured from the resolved key."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=_load_openai_key(rt))
    print("OpenAI client initialized")
    return client


def _load_hf_token(rt: RuntimeInfo) -> str:
    """Resolve HF token: Colab userdata > env > HF_key.txt at experiment root."""
    if rt.in_colab:
        try:
            from google.colab import userdata  # type: ignore
            tok = userdata.get("huggingface")
            if tok:
                return str(tok).strip()
        except Exception:
            pass

    tok = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if tok:
        return tok

    kp = os.path.join(rt.experiment_root, "HF_key.txt")
    if os.path.exists(kp):
        return Path(kp).read_text(encoding="utf-8").strip()
    return ""


def authenticate(rt: RuntimeInfo, hf: bool = True, wandb_enabled: bool = False) -> None:
    """Log in to Hugging Face (always) and W&B (Colab only, optional).

    On both hosts the HF token is resolved via :func:`_load_hf_token`
    (Colab userdata → env → ``HF_key.txt``). On local the W&B login is
    skipped — set ``WANDB_API_KEY`` in your shell or pass ``report_to=[]``
    if you don't want W&B.
    """
    if hf:
        token = _load_hf_token(rt)
        if token:
            from huggingface_hub import login
            login(token=token)
            print("Logged in to Hugging Face Hub")
        else:
            print("WARNING: No HF token found (looked in Colab userdata, env, HF_key.txt).")

    if wandb_enabled and rt.in_colab:
        from google.colab import userdata  # type: ignore
        import wandb
        wb_key = userdata.get("wandb")
        if wb_key:
            wandb.login(key=wb_key)
            print("Logged in to Weights & Biases")
        else:
            print("WARNING: REPORT_TO includes wandb but no Colab userdata key named 'wandb'.")
    elif wandb_enabled:
        print("Local W&B: ensure WANDB_API_KEY is set in your shell (or remove 'wandb' from REPORT_TO).")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              PREFLIGHT                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def verify_helpers(modules: List[str], paths: List[str]) -> None:
    """Import-test each module and ensure each path is writable.

    Raises ``RuntimeError`` if any module fails to import. Creates paths if missing.
    """
    print("=" * 70)
    print("RUNTIME PREFLIGHT CHECK")
    print("=" * 70)

    missing: List[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"Import OK: {mod}")
        except Exception as e:
            print(f"Import failed: {mod} -> {e}")
            missing.append(mod)

    for path_str in paths:
        p = Path(path_str)
        p.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {p}")

    if paths:
        probe = Path(paths[0]) / ".write_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        print(f"Write test OK: {probe.parent}")

    if missing:
        print("\nPreflight failed: missing imports ->", missing)
        raise RuntimeError("Preflight failed due to missing helper imports.")
    print("\nPreflight passed: imports + folders + write access are ready.")
    print("=" * 70)
