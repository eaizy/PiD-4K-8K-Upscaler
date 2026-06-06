"""
Project & file storage — the FastAPI replacement for the Tauri IPC commands.

The original Uniqlaw integration kept projects on disk and reached them through
Tauri commands (`pid_save_manifest`, `pid_save_image`, `pid_list_projects`, …).
This module reimplements those operations as plain functions over the local
filesystem; server.py exposes them as HTTP endpoints so the web frontend can
call them with `fetch` instead of `invoke`.

Disk layout (under PID_DATA_ROOT):
    <slug>/
      manifest.json
      input.{png,jpg,webp}
      output_{epoch}.png
      job_{jobId}/ ...            (working dirs created by the job runner)
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import config

_IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")
_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}
_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def sanitize_slug(slug: str) -> str:
    """Project slug → safe folder name ([a-z0-9-] only). Guards against path
    traversal (no '..', no separators leak through)."""
    s = _SLUG_RE.sub("-", (slug or "").strip().lower()).strip("-")
    return s or f"pid-{int(time.time())}"


def project_dir(slug: str, *, create: bool = False) -> Path:
    d = config.PID_DATA_ROOT / sanitize_slug(slug)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ── Manifest ────────────────────────────────────────────────────────────────
def manifest_path(slug: str) -> Path:
    return project_dir(slug) / "manifest.json"


def save_manifest(slug: str, manifest: dict) -> str:
    d = project_dir(slug, create=True)
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def load_manifest(slug: str) -> Optional[dict]:
    p = manifest_path(slug)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_project(title: str, slug: Optional[str] = None) -> dict:
    final_slug = sanitize_slug(slug) if slug else f"pid-{int(time.time()*1000):x}"
    now = int(time.time() * 1000)
    manifest = {
        "slug": final_slug,
        "title": (title or "").strip() or "Untitled PiD Project",
        "status": "idle",
        "inputKey": None,
        "outputs": [],
        "params": {
            "targetQuality": "4k",
            "inferenceSteps": 4,
            "degradeSigma": 0,
            "saveIntermediate": True,
        },
        "createdAt": now,
        "updatedAt": now,
    }
    save_manifest(final_slug, manifest)
    return manifest


def list_projects() -> list[dict]:
    root = config.PID_DATA_ROOT
    if not root.is_dir():
        return []
    out: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = load_manifest(d.name)
        if not m:
            continue
        out.append(
            {
                "slug": m.get("slug", d.name),
                "title": m.get("title", d.name),
                "updatedAt": m.get("updatedAt", 0),
                "outputCount": len(m.get("outputs", []) or []),
            }
        )
    out.sort(key=lambda r: r.get("updatedAt", 0), reverse=True)
    return out


def delete_project(slug: str) -> None:
    d = project_dir(slug)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


# ── Images ──────────────────────────────────────────────────────────────────
def find_image(slug: str, key: str) -> Optional[Path]:
    """Find <slug>/<key>.{png,jpg,jpeg,webp}."""
    d = project_dir(slug)
    safe_key = Path(key).name  # no traversal
    for ext in _IMAGE_EXTS:
        p = d / f"{safe_key}.{ext}"
        if p.is_file():
            return p
    return None


def image_mime(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower().lstrip("."), "application/octet-stream")


def save_image_base64(slug: str, key: str, base64_data: str, mime: str) -> Path:
    d = project_dir(slug, create=True)
    ext = _EXT_BY_MIME.get(mime, "png")
    safe_key = Path(key).name
    # remove any existing variants of this key so the extension stays consistent
    for e in _IMAGE_EXTS:
        old = d / f"{safe_key}.{e}"
        if old.is_file():
            old.unlink()
    raw = base64.b64decode(base64_data)
    p = d / f"{safe_key}.{ext}"
    p.write_bytes(raw)
    return p


def save_image_bytes(slug: str, key: str, raw: bytes, filename: str) -> Path:
    d = project_dir(slug, create=True)
    ext = Path(filename).suffix.lower().lstrip(".") or "png"
    if ext not in _IMAGE_EXTS:
        ext = "png"
    safe_key = Path(key).name
    for e in _IMAGE_EXTS:
        old = d / f"{safe_key}.{e}"
        if old.is_file():
            old.unlink()
    p = d / f"{safe_key}.{ext}"
    p.write_bytes(raw)
    return p


def delete_image(slug: str, key: str) -> None:
    d = project_dir(slug)
    safe_key = Path(key).name
    for ext in _IMAGE_EXTS:
        p = d / f"{safe_key}.{ext}"
        if p.is_file():
            p.unlink()


def image_dimensions(path: Path) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


# ── Model status (replaces pid_check_models) ─────────────────────────────────
def check_models() -> dict:
    """Mirror of the original PidModelStatus, computed from config paths."""
    ckpt2k = (
        config.PID_MODELS_ROOT
        / "checkpoints"
        / "PiD_res2k_sr4x_official_flux_distill_4step"
        / "model_ema_bf16.pth"
    )
    ckpt4k = (
        config.PID_MODELS_ROOT
        / "checkpoints"
        / "PiD_res2kto4k_sr4x_official_flux_distill_4step"
        / "model_ema_bf16.pth"
    )
    return {
        "modelsRoot": str(config.PID_MODELS_ROOT),
        "has2k": ckpt2k.is_file(),
        "has4k": ckpt4k.is_file(),
        "fluxVaePath": str(config.PID_FLUX_VAE) if config.PID_FLUX_VAE.is_file() else None,
        "downloadedBytes": 0,
    }
