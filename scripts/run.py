#!/usr/bin/env python3
"""
PiD Studio — launch backend + frontend together.

    python scripts/run.py            # dev mode: FastAPI + Vite dev server
    python scripts/run.py --no-open  # don't auto-open the browser

Starts the FastAPI backend (uvicorn) and the Vite dev server, waits for the
frontend to come up, and opens the browser. Ctrl-C stops both.

The in-app setup wizard handles installing ComfyUI + downloading model weights
on first run (it talks to the backend's /setup endpoints), so you don't need to
run anything else — just provide HF_TOKEN in .env.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"
FRONTEND_URL = "http://localhost:5173"


def venv_python() -> str:
    p = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return str(p) if p.exists() else sys.executable


def main() -> int:
    no_open = "--no-open" in sys.argv
    procs: list[subprocess.Popen] = []

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("==> Starting backend (FastAPI) on :17820")
    procs.append(subprocess.Popen([venv_python(), "server.py"], cwd=str(REPO / "backend"), env=env))

    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    if (REPO / "frontend" / "node_modules").exists():
        print("==> Starting frontend (Vite) on :5173")
        procs.append(subprocess.Popen([npm, "run", "dev"], cwd=str(REPO / "frontend"), env=env))
    else:
        print("!!  frontend/node_modules missing — run `python scripts/bootstrap.py` first.")

    if not no_open and len(procs) > 1:
        time.sleep(4)
        webbrowser.open(FRONTEND_URL)

    print("\nPiD Studio is starting. Press Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() is not None:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n==> Shutting down…")
        for p in procs:
            try:
                if sys.platform == "win32":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGINT)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=8)
            except Exception:
                p.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
