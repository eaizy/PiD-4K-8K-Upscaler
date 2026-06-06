"""
Shared test fixtures.

These tests run against a LIVE backend (the one you already run locally, or the
ported server.py started with `python scripts/run.py`). Point them elsewhere
with PID_BASE, e.g. PID_BASE=http://127.0.0.1:17820.

Tests that need ComfyUI + the model weights (the real upscale) auto-skip when
those aren't ready, so the suite stays green on a machine without a GPU.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import pytest

# Make the backend modules (config, comfy_client, comfy_workflow, projects)
# importable from tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("PID_BASE", "http://127.0.0.1:17820").rstrip("/")
WS_BASE = BASE.replace("http://", "ws://").replace("https://", "wss://")


def _get_json(path: str, timeout: float = 10.0):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def backend_up() -> bool:
    try:
        _get_json("/health", timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def base() -> str:
    return BASE


@pytest.fixture(scope="session")
def ws_base() -> str:
    return WS_BASE


@pytest.fixture(scope="session")
def health() -> dict:
    if not backend_up():
        pytest.skip(f"backend not reachable at {BASE} (start it first)")
    return _get_json("/health")


@pytest.fixture(autouse=True)
def _require_backend(request):
    """Skip every test in this suite if the backend is down."""
    if not backend_up():
        pytest.skip(f"backend not reachable at {BASE}")
