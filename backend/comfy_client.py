"""
ComfyUI HTTP API client (async).

Runs the PiD upscale through ComfyUI's native node graph:
  1. upload_image    → upload the input PNG to ComfyUI
  2. submit_prompt   → POST the workflow to /prompt, get a prompt_id
  3. stream_execution→ listen on /ws for real step progress + latent2rgb preview
                       frames (falls back to /history polling if WS is absent)
  4. fetch_image     → fetch the output PNG bytes via /view

ComfyUI default port 8188. Uses only the stdlib (urllib) for HTTP; `websockets`
is optional and enables the live preview stream when present.
"""

from __future__ import annotations

import asyncio
import json
import struct
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import websockets  # type: ignore

    _HAS_WS = True
except Exception:  # pragma: no cover
    _HAS_WS = False


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188) -> None:
        self.base = f"http://{host}:{port}"

    # ---- low-level HTTP (blocking, wrapped with asyncio.to_thread) ----
    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.read()

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _upload_multipart(self, file_path: Path, subfolder: str = "pid") -> dict:
        """POST /upload/image (multipart/form-data)."""
        boundary = "----PiDStudioBoundary7MA4YWxkTrZu0gW"
        body = bytearray()
        # file field
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="image"; filename="{file_path.name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
        # subfolder + overwrite fields
        for k, v in (("subfolder", subfolder), ("overwrite", "true")):
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
            body.extend(f"{v}\r\n".encode())
        body.extend(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            self.base + "/upload/image",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    # ---- async public API ----
    async def health(self) -> bool:
        try:
            await asyncio.to_thread(self._get, "/system_stats")
            return True
        except Exception:
            return False

    async def object_info(self, node: str) -> Optional[dict]:
        try:
            raw = await asyncio.to_thread(self._get, f"/object_info/{node}")
            return json.loads(raw.decode("utf-8")).get(node)
        except Exception:
            return None

    async def upload_image(self, file_path: Path) -> str:
        """Upload the input image to ComfyUI's input/ folder. Returns the
        'subfolder/filename' reference that LoadImage expects."""
        res = await asyncio.to_thread(self._upload_multipart, file_path)
        name = res.get("name", file_path.name)
        sub = res.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    async def submit_prompt(self, workflow: dict, client_id: str) -> str:
        res = await asyncio.to_thread(
            self._post_json, "/prompt", {"prompt": workflow, "client_id": client_id}
        )
        pid = res.get("prompt_id")
        if not pid:
            raise RuntimeError(f"ComfyUI /prompt rejected: {res}")
        return pid

    async def wait_for_result(
        self,
        prompt_id: str,
        on_progress: Optional[Callable[[float, str], None]] = None,
        timeout_s: float = 600.0,
        poll_interval: float = 1.0,
    ) -> dict:
        """Polls /history/<prompt_id> until completion, then returns the history
        entry. Raises RuntimeError on error/timeout.

        ComfyUI /prompt is async — completion shows up in /history. Progress is
        a time-based heuristic here (the WS path is more granular; PiD 4-step is
        fast enough that polling is fine)."""
        loop = asyncio.get_event_loop()
        start = loop.time()
        last_prog = 0.0
        while loop.time() - start < timeout_s:
            try:
                raw = await asyncio.to_thread(self._get, f"/history/{prompt_id}")
                hist = json.loads(raw.decode("utf-8"))
            except Exception:
                hist = {}
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    if on_progress:
                        on_progress(1.0, "saving")
                    return entry
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution error: {msgs}")
            if on_progress:
                elapsed = loop.time() - start
                est = min(0.9, elapsed / 20.0)
                if est > last_prog:
                    last_prog = est
                    on_progress(est, "running")
            await asyncio.sleep(poll_interval)
        raise RuntimeError(f"ComfyUI prompt {prompt_id} timed out after {timeout_s}s")

    async def stream_execution(
        self,
        prompt_id: str,
        client_id: str,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_preview: Optional[Callable[[bytes], None]] = None,
        timeout_s: float = 600.0,
    ) -> dict:
        """Connects to ComfyUI /ws and streams REAL step progress + latent2rgb
        preview frames. Returns the /history entry on completion.

        Falls back to poll-based wait_for_result if websockets is unavailable or
        the connection drops (no regression). Binary message format:
        [4B event_type][4B image_type][image bytes]; event_type==1 → preview."""
        if not _HAS_WS:
            return await self.wait_for_result(prompt_id, on_progress, timeout_s)

        ws_url = self.base.replace("http://", "ws://") + f"/ws?clientId={client_id}"
        try:
            async with websockets.connect(
                ws_url, max_size=None, ping_interval=None
            ) as ws:
                loop = asyncio.get_event_loop()
                start = loop.time()
                while loop.time() - start < timeout_s:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                    if isinstance(raw, (bytes, bytearray)):
                        if on_preview and len(raw) > 8:
                            ev_type = struct.unpack(">I", raw[:4])[0]
                            if ev_type == 1:  # preview image
                                on_preview(bytes(raw[8:]))
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    data = msg.get("data", {}) or {}
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    if mtype == "progress":
                        val = float(data.get("value", 0))
                        mx = float(data.get("max", 1)) or 1.0
                        if on_progress:
                            on_progress(min(0.99, val / mx), "running")
                    elif mtype == "execution_error":
                        raise RuntimeError(f"ComfyUI execution error: {data}")
                    elif (
                        mtype == "executing"
                        and data.get("node") is None
                        and data.get("prompt_id") == prompt_id
                    ):
                        break  # execution complete
                if on_progress:
                    on_progress(1.0, "saving")
        except RuntimeError:
            raise
        except Exception:
            # WS dropped/failed → poll fallback.
            return await self.wait_for_result(prompt_id, on_progress, timeout_s)

        # Short poll so SaveImage finishes writing to disk.
        loop = asyncio.get_event_loop()
        start = loop.time()
        while loop.time() - start < 30.0:
            try:
                raw = await asyncio.to_thread(self._get, f"/history/{prompt_id}")
                entry = json.loads(raw.decode("utf-8")).get(prompt_id)
                if entry and entry.get("outputs"):
                    return entry
            except Exception:
                pass
            await asyncio.sleep(0.3)
        raise RuntimeError(f"ComfyUI prompt {prompt_id} completed but history empty")

    def extract_output(self, history_entry: dict, save_node: str = "save") -> Optional[dict]:
        """First image meta {filename, subfolder, type} from the SaveImage node."""
        outputs = history_entry.get("outputs", {})
        node_out = outputs.get(save_node) or {}
        images = node_out.get("images") or []
        if images:
            return images[0]
        for v in outputs.values():
            imgs = v.get("images") if isinstance(v, dict) else None
            if imgs:
                return imgs[0]
        return None

    async def fetch_image(self, image_meta: dict) -> bytes:
        q = urllib.parse.urlencode(
            {
                "filename": image_meta.get("filename", ""),
                "subfolder": image_meta.get("subfolder", ""),
                "type": image_meta.get("type", "output"),
            }
        )
        return await asyncio.to_thread(self._get, f"/view?{q}")
