#!/usr/bin/env python3
"""
PiD Studio — one-command bootstrap.

    python scripts/bootstrap.py

Sets up *code* dependencies only (fast, no model downloads):
  1. Verifies Python >= 3.10.
  2. Creates a backend virtualenv at .venv and installs backend/requirements.txt.
  3. Installs frontend npm dependencies (if npm is available).
  4. Copies .env.example → .env if missing, and reminds you to set HF_TOKEN.

The heavy parts (ComfyUI + multi-GB model weights) are installed on first launch
by the in-app setup wizard, or run `python scripts/run.py` and follow the prompt.
They are deliberately NOT done here so bootstrap stays quick and offline-friendly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"


def step(msg: str) -> None:
    print(f"\n==> {msg}")


def run(args: list[str], cwd: Path | None = None) -> int:
    print("    $", " ".join(args))
    return subprocess.call(args, cwd=str(cwd) if cwd else None)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    if sys.version_info < (3, 10):
        print(f"Python 3.10+ required, found {sys.version.split()[0]}")
        return 1

    step("Creating backend virtualenv (.venv)")
    if not VENV.exists():
        if run([sys.executable, "-m", "venv", str(VENV)]) != 0:
            print("venv creation failed")
            return 1
    py = str(venv_python())

    step("Installing backend dependencies")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    if run([py, "-m", "pip", "install", "-r", str(REPO / "backend" / "requirements.txt")]) != 0:
        print("backend dependency install failed")
        return 1

    step("Installing frontend dependencies")
    npm = shutil.which("npm")
    if npm:
        run([npm, "install"], cwd=REPO / "frontend")
    else:
        print("    npm not found — skipping frontend deps (install Node.js to build the UI).")

    step("Environment file")
    env = REPO / ".env"
    if not env.exists():
        shutil.copy2(REPO / ".env.example", env)
        print("    Created .env from .env.example.")
    print("    → Edit .env and set HF_TOKEN (https://huggingface.co/settings/tokens).")
    print("    → Accept the license at https://huggingface.co/nvidia/PiD (gated, NSCLv1).")

    print("\nDone. Next:")
    print("    python scripts/run.py        # starts backend + frontend, opens the app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
