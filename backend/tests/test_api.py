"""
Project + file API tests — the endpoints that replaced the Tauri IPC commands.

These need only a running backend (no GPU / ComfyUI / model weights), so they
verify the core contract the frontend depends on: create → upload → read → list
→ delete a project, plus the /health and /models shapes.
"""

from __future__ import annotations

import io
import json
import urllib.request
import uuid

import pytest

from conftest import BASE


def _req(method: str, path: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read()
        return r.status, (json.loads(body) if body else None)


def _post_json(path: str, payload: dict):
    return _req("POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"})


def _png_bytes(size: int = 64) -> bytes:
    from PIL import Image
    import numpy as np

    arr = np.zeros((size, size, 3), dtype="uint8")
    grad = np.linspace(0, 255, size, dtype="uint8")
    arr[:, :, 0] = grad[None, :]
    arr[:, :, 1] = grad[:, None]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health_shape(health):
    for key in ("status", "comfyReady", "jobsInQueue"):
        assert key in health


def test_models_endpoint():
    status, body = _req("GET", "/models")
    assert status == 200
    assert "modelsRoot" in body and "has2k" in body


def test_project_lifecycle():
    title = f"pytest-{uuid.uuid4().hex[:6]}"
    status, manifest = _post_json("/projects", {"title": title})
    assert status == 200
    slug = manifest["slug"]
    try:
        assert manifest["title"] == title
        assert manifest["outputs"] == []

        # upload an input image (multipart)
        png = _png_bytes()
        boundary = "----pytestBoundary"
        body = bytearray()
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="key"\r\n\r\ninput\r\n'
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="file"; filename="input.png"\r\n'
        body += b"Content-Type: image/png\r\n\r\n"
        body += png + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        status, up = _req(
            "POST", f"/projects/{slug}/images", bytes(body),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert status == 200
        assert up["resolution"] == [64, 64]

        # the image is served back
        with urllib.request.urlopen(f"{BASE}/projects/{slug}/images/input", timeout=10) as r:
            assert r.status == 200
            assert r.read()[:8] == b"\x89PNG\r\n\x1a\n"

        # appears in the project list
        _, projects = _req("GET", "/projects")
        assert any(p["slug"] == slug for p in projects)

        # manifest round-trips through PUT
        manifest["inputKey"] = "input"
        status, _ = _req(
            "PUT", f"/projects/{slug}", json.dumps(manifest).encode(),
            {"Content-Type": "application/json"},
        )
        assert status == 200
        _, reloaded = _req("GET", f"/projects/{slug}")
        assert reloaded["inputKey"] == "input"
    finally:
        _req("DELETE", f"/projects/{slug}")
        with pytest.raises(urllib.error.HTTPError):
            _req("GET", f"/projects/{slug}")
