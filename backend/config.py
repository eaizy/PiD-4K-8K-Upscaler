"""
PiD Studio — central configuration.

Every path and port the backend needs is resolved here, from environment
variables with self-contained defaults. This is the file that replaces the
hard-coded ``C:/AI/...`` constants from the original Uniqlaw Tauri sidecar so
that a fresh fork works cross-platform with zero manual path editing.

Layout (defaults, all under the repo so a clone "just works"):

    <repo>/
      .pidstudio/            PID_HOME      — everything installed at runtime
        ComfyUI/             COMFY_ROOT    — auto-installed ComfyUI
          models/            COMFY_MODELS  — diffusion_models/, vae/, clip/, ...
        models/              PID_MODELS    — subprocess (scale_rae) .pth checkpoints
        pid-upstream/        PID_REPO_DIR  — cloned nv-tlabs/PiD
      data/                  PID_DATA_ROOT — user projects (manifest.json + images)

Override any of these via the matching environment variable (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env from the repo root if python-dotenv is available (optional dep).
REPO_ROOT = Path(__file__).resolve().parent.parent
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass


def _path_env(name: str, default: Path) -> Path:
    """Read a path from env, falling back to ``default``. Empty string = default."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ── Install roots ─────────────────────────────────────────────────────────
PID_HOME = _path_env("PID_HOME", REPO_ROOT / ".pidstudio")
COMFY_ROOT = _path_env("COMFY_ROOT", PID_HOME / "ComfyUI")
COMFY_MODELS_ROOT = _path_env("COMFY_MODELS_ROOT", COMFY_ROOT / "models")
PID_MODELS_ROOT = _path_env("PID_MODELS_ROOT", PID_HOME / "models")
PID_REPO_DIR = _path_env("PID_REPO_DIR", PID_HOME / "pid-upstream")
PID_DATA_ROOT = _path_env("PID_DATA_ROOT", REPO_ROOT / "data")

# Flux 16-ch VAE used by the subprocess (scale_rae) path; ComfyUI flux/flux2
# paths use their own VAE files referenced in comfy_workflow.PID_ENCODE_VAE.
PID_FLUX_VAE = _path_env("PID_FLUX_VAE", COMFY_MODELS_ROOT / "vae" / "ae.safetensors")
PID_FLUX2_VAE = PID_MODELS_ROOT / "checkpoints" / "flux2_ae.safetensors"

# Python interpreter used to launch the PiD reference CLI subprocess (scale_rae).
# Resolution order: explicit PID_PYTHON env → the dedicated pid-venv the installer
# builds at <PID_HOME>/pid-venv (torch + editable pid-upstream) → this interpreter.
import sys as _sys

PID_VENV_DIR = PID_HOME / "pid-venv"
PID_VENV_PYTHON = PID_VENV_DIR / (
    "Scripts/python.exe" if _sys.platform == "win32" else "bin/python"
)


def _default_pid_python() -> str:
    if PID_VENV_PYTHON.exists():
        return str(PID_VENV_PYTHON)
    return _sys.executable


PID_PYTHON = os.environ.get("PID_PYTHON", "").strip() or _default_pid_python()

# Torch index for the pid-venv (CUDA build). Override for CPU/ROCm/other CUDA.
PID_TORCH_INDEX_URL = os.environ.get(
    "PID_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124"
)

# ── Ports / network ───────────────────────────────────────────────────────
PORT = int(os.environ.get("PID_PORT", "17820"))
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))

# ── Secrets / behavior ────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
AUTO_INSTALL = _bool_env("PID_AUTO_INSTALL", True)

# Hugging Face repo that hosts the PiD weights (gated, NSCLv1 non-commercial).
HF_PID_REPO = os.environ.get("HF_PID_REPO", "nvidia/PiD")


def ensure_runtime_dirs() -> None:
    """Create the directories the backend writes into. Install targets
    (ComfyUI, models) are created by the installer, not here."""
    PID_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PID_HOME.mkdir(parents=True, exist_ok=True)


def summary() -> dict:
    """Resolved config, for /health and debugging (no secrets)."""
    return {
        "repoRoot": str(REPO_ROOT),
        "pidHome": str(PID_HOME),
        "comfyRoot": str(COMFY_ROOT),
        "comfyModelsRoot": str(COMFY_MODELS_ROOT),
        "pidModelsRoot": str(PID_MODELS_ROOT),
        "pidRepoDir": str(PID_REPO_DIR),
        "pidDataRoot": str(PID_DATA_ROOT),
        "port": PORT,
        "comfy": f"{COMFY_HOST}:{COMFY_PORT}",
        "hfTokenPresent": bool(HF_TOKEN),
        "autoInstall": AUTO_INSTALL,
    }
