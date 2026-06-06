"""
Full HTTP + WebSocket E2E: create project → upload → POST /upscale → listen on
/preview → assert a non-black output is produced and served.

This is the exact flow the frontend drives. It exercises the ComfyUI-backed
flux/flux2 path end to end. Auto-skips unless ComfyUI is online and the model
for the chosen backbone/quality is present.

Ported from the original sidecar _http_e2e_test.py.

    pytest backend/tests/test_http_e2e.py
    PID_E2E_QUALITY=4k PID_E2E_BACKBONE=flux2 pytest backend/tests/test_http_e2e.py
"""

from __future__ import annotations

import io
import json
import os
import urllib.request

import pytest

from conftest import BASE, WS_BASE

QUALITY = os.environ.get("PID_E2E_QUALITY", "2k")
BACKBONE = os.environ.get("PID_E2E_BACKBONE", "flux")

try:
    from websockets.sync.client import connect as ws_connect
except Exception:  # pragma: no cover
    ws_connect = None


def _model_ready(health: dict) -> bool:
    if BACKBONE == "scale_rae":
        return bool(health.get("scaleRaeReady"))
    if not health.get("comfyReady"):
        return False
    flag = {
        ("flux", "2k"): "comfyFlux2kReady",
        ("flux", "4k"): "comfyFlux4kReady",
        ("flux2", "2k"): "comfyFlux22kReady",
        ("flux2", "4k"): "comfyFlux24kReady",
    }[(BACKBONE, "2k" if QUALITY == "2k" else "4k")]
    return bool(health.get(flag))


def _make_input_png(size: int = 768) -> bytes:
    from PIL import Image
    import numpy as np

    arr = np.zeros((size, size, 3), dtype="uint8")
    grad = np.linspace(0, 255, size, dtype="uint8")
    arr[:, :, 0] = grad[None, :]
    arr[:, :, 1] = grad[:, None]
    arr[::32, :, 2] = 255  # horizontal lines (detail probe)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _upload(slug: str, png: bytes) -> None:
    boundary = "----e2eBoundary"
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="key"\r\n\r\ninput\r\n'
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="input.png"\r\n'
    body += b"Content-Type: image/png\r\n\r\n"
    body += png + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/projects/{slug}/images", data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    urllib.request.urlopen(req, timeout=30).read()


def _post_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def test_full_upscale_roundtrip(health):
    if ws_connect is None:
        pytest.skip("websockets sync client unavailable")
    if not _model_ready(health):
        pytest.skip(f"{BACKBONE}/{QUALITY} not ready (ComfyUI offline or model missing)")

    import numpy as np
    from PIL import Image

    # create project + input
    manifest = _post_json("/projects", {"title": "e2e-http"})
    slug = manifest["slug"]
    try:
        _upload(slug, _make_input_png(768))
        manifest["inputKey"] = "input"
        urllib.request.urlopen(
            urllib.request.Request(
                f"{BASE}/projects/{slug}", data=json.dumps(manifest).encode(),
                method="PUT", headers={"Content-Type": "application/json"},
            ),
            timeout=15,
        ).read()

        # connect WS BEFORE enqueuing (don't miss the queued event)
        done_evt = None
        with ws_connect(f"{WS_BASE}/preview") as ws:
            res = _post_json("/upscale", {
                "slug": slug, "inputKey": "input", "backbone": BACKBONE,
                "targetQuality": QUALITY, "inferenceSteps": 4, "degradeSigma": 0.0,
                "saveIntermediate": True, "seed": 42, "prompt": None,
            })
            job_id = res["jobId"]
            previews = 0
            while True:
                raw = ws.recv(timeout=180)
                evt = json.loads(raw)
                if evt.get("jobId") != job_id:
                    continue
                if evt.get("type") == "xt-step":
                    previews += 1
                elif evt.get("type") == "done":
                    done_evt = evt
                    break
                elif evt.get("type") == "error":
                    pytest.fail(f"upscale error: {evt.get('message')}")

        assert done_evt is not None
        out_key = done_evt["outputKey"]

        # output is served and not flat/black
        with urllib.request.urlopen(f"{BASE}/projects/{slug}/images/{out_key}", timeout=20) as r:
            out_bytes = r.read()
        arr = np.asarray(Image.open(io.BytesIO(out_bytes)).convert("RGB"))
        assert arr.std() >= 1.0, "output is flat/black (NaN suspected)"
    finally:
        urllib.request.urlopen(
            urllib.request.Request(f"{BASE}/projects/{slug}", method="DELETE"), timeout=15
        ).read()
