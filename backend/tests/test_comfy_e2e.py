"""
ComfyUI PiD node-graph smoke test (ported from _comfy_e2e_test.py).

Talks directly to ComfyUI (not through the backend job queue):
  1. verify the required node schemas exist via /object_info
  2. upload a test image, build_workflow → /prompt, poll /history
  3. fetch the output and assert it isn't flat/black

Auto-skips when ComfyUI isn't reachable.

    pytest backend/tests/test_comfy_e2e.py
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid

import pytest

import config
from comfy_client import ComfyClient
from comfy_workflow import build_workflow, sigmas_for_steps, target_dims

QUALITY = os.environ.get("PID_E2E_QUALITY", "2k")
BACKBONE = os.environ.get("PID_E2E_BACKBONE", "flux")

REQUIRED_NODES = [
    "VAELoader", "LoadImage", "VAEEncode", "CLIPLoader", "CLIPTextEncode",
    "ConditioningZeroOut", "PiDConditioning", "UNETLoader",
    "EmptyChromaRadianceLatentImage", "KSamplerSelect", "ManualSigmas",
    "SamplerCustom", "VAEDecode", "SaveImage",
]


def _comfy() -> ComfyClient:
    return ComfyClient(host=config.COMFY_HOST, port=config.COMFY_PORT)


def test_required_nodes_present():
    async def run():
        c = _comfy()
        if not await c.health():
            pytest.skip("ComfyUI not reachable")
        missing = [n for n in REQUIRED_NODES if await c.object_info(n) is None]
        assert not missing, f"missing ComfyUI nodes: {missing}"

    asyncio.run(run())


def test_pure_upscale_workflow():
    import numpy as np
    from PIL import Image

    async def run():
        c = _comfy()
        if not await c.health():
            pytest.skip("ComfyUI not reachable")

        in_size = 1024 if QUALITY == "4k" else 512
        work = config.PID_DATA_ROOT / "_comfy_e2e"
        work.mkdir(parents=True, exist_ok=True)
        src = work / f"input_{in_size}.png"
        arr = np.zeros((in_size, in_size, 3), dtype="uint8")
        grad = np.linspace(0, 255, in_size, dtype="uint8")
        arr[:, :, 0] = grad[None, :]
        arr[:, :, 1] = grad[:, None]
        Image.fromarray(arr).save(src)

        ref = await c.upload_image(src)
        out_w, out_h = target_dims(QUALITY, 1.0)
        wf = build_workflow(
            image_filename=ref, backbone=BACKBONE, target_quality=QUALITY,
            out_w=out_w, out_h=out_h, seed=42, prompt=None, degrade_sigma=0.0,
            filename_prefix=f"pidstudio_e2e_{QUALITY}", inference_steps=4,
        )
        assert sigmas_for_steps(4) == "0.999,0.866,0.634,0.342,0"

        try:
            prompt_id = await c.submit_prompt(wf, uuid.uuid4().hex)
        except Exception as e:
            pytest.skip(f"ComfyUI rejected the workflow (models likely missing): {e}")

        entry = await c.wait_for_result(prompt_id, timeout_s=300)
        meta = c.extract_output(entry)
        assert meta, f"no output image. outputs={list(entry.get('outputs', {}).keys())}"
        png = await c.fetch_image(meta)
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        assert a.std() >= 1.0, "output flat/black (NaN suspected)"

    asyncio.run(run())
