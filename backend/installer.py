"""
Auto-installer — "fork, set HF_TOKEN, run".

Goal: a person who clones this repo provides exactly one secret (HF_TOKEN) and
the app pulls everything else into the right place:

  * ComfyUI            → cloned to COMFY_ROOT (recent ComfyUI ships the PiD
                          nodes in core via PR comfyanonymous/ComfyUI#14103, so
                          no custom node is needed on a fresh clone).
  * PiD model weights  → downloaded from Hugging Face into ComfyUI's model dirs,
                          under the exact filenames comfy_workflow.py expects.
  * pid-upstream repo  → cloned for the scale_rae (8×) subprocess backbone.

Progress is streamed to the frontend over the same /preview WebSocket as
upscale jobs, as {"type": "setup", ...} events, so the first-run setup wizard
can show a live log + progress bar.

VERIFIED model sources (Hugging Face):
  - Comfy-Org/PixelDiT : the ComfyUI-format PiD diffusion models + gemma encoder
                          (filenames match comfy_workflow.PID_MODEL_FILES exactly).
  - nvidia/PiD         : the reference .pth checkpoints used by the subprocess
                          backbone (scale_rae) + flux2_ae VAE. NSCLv1, gated.

The Flux/Flux2 VAE files used for VAEEncode are environment-specific names and
their canonical source is not pinned here; see VAE_MANIFEST + docs/INSTALL.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger("pid-studio.installer")

COMFYUI_GIT = "https://github.com/comfyanonymous/ComfyUI"
PID_UPSTREAM_GIT = "https://github.com/nv-tlabs/PiD"

# Scale-RAE (8× single-pass) subprocess backbone: gated nvidia/PiD checkpoint.
SCALE_RAE_REPO = config.HF_PID_REPO  # nvidia/PiD
SCALE_RAE_CKPT_REL = (
    "checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth"
)


# ── Model manifest ───────────────────────────────────────────────────────────
@dataclass
class ModelFile:
    repo: str            # HF repo id
    path: str            # path within the repo
    dest_subdir: str     # subdir under COMFY_MODELS_ROOT
    dest_name: str       # final filename (must match comfy_workflow.py)
    verified: bool = True
    gated: bool = False  # requires HF_TOKEN + accepted license


# ComfyUI-format diffusion models + text encoder (Comfy-Org/PixelDiT, public).
# Names verified against https://huggingface.co/Comfy-Org/PixelDiT/tree/main and
# they match comfy_workflow.PID_MODEL_FILES / PID_CLIP_NAME exactly.
COMFY_MODELS: list[ModelFile] = [
    ModelFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux1_512_to_2048_4step_bf16.safetensors",
              "diffusion_models", "pid_flux1_512_to_2048_4step_bf16.safetensors"),
    ModelFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux1_1024_to_4096_4step_bf16.safetensors",
              "diffusion_models", "pid_flux1_1024_to_4096_4step_bf16.safetensors"),
    ModelFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux2_512_to_2048_4step_bf16.safetensors",
              "diffusion_models", "pid_flux2_512_to_2048_4step_bf16.safetensors"),
    ModelFile("Comfy-Org/PixelDiT", "diffusion_models/pid_flux2_1024_to_4096_4step_bf16.safetensors",
              "diffusion_models", "pid_flux2_1024_to_4096_4step_bf16.safetensors"),
    ModelFile("Comfy-Org/PixelDiT", "text_encoders/gemma_2_2b_it_elm_bf16.safetensors",
              "text_encoders", "gemma_2_2b_it_elm_bf16.safetensors"),
]

# VAE files used for VAEEncode (flux = 16-ch, flux2 = 128-ch). Sources pinned to
# the public Comfy/Kijai repackages whose contents match the filenames the
# workflow expects. Override with PID_FLUX_VAE_REPO/PATH (and FLUX2_*) if you
# host them elsewhere. The pixel_space DECODE VAE needs no file (core PiD node).
VAE_MANIFEST: list[ModelFile] = [
    ModelFile(
        os.environ.get("PID_FLUX_VAE_REPO", "Kijai/flux-fp8"),
        os.environ.get("PID_FLUX_VAE_PATH", "flux-vae-bf16.safetensors"),
        "vae", "flux-vae-bf16.safetensors",
    ),
    ModelFile(
        os.environ.get("PID_FLUX2_VAE_REPO", "Comfy-Org/flux2-dev"),
        os.environ.get("PID_FLUX2_VAE_PATH", "split_files/vae/flux2-vae.safetensors"),
        "vae", "flux2-vae.safetensors",
    ),
]


def _emit(bus, step: str, status: str, message: str = "", progress: float = 0.0) -> None:
    """Publish a setup event onto the shared event bus (frontend listens on WS)."""
    log.info("[setup] %-14s %-8s %s", step, status, message)
    if bus is not None:
        bus.publish({
            "type": "setup", "step": step, "status": status,
            "message": message, "progress01": round(progress, 3),
        })


# ── Status (what's installed vs missing) ─────────────────────────────────────
def _model_present(mf: ModelFile) -> bool:
    return (config.COMFY_MODELS_ROOT / mf.dest_subdir / mf.dest_name).is_file()


async def status(comfy) -> dict:
    comfy_up = await comfy.health()
    comfy_models = [
        {"name": mf.dest_name, "subdir": mf.dest_subdir, "present": _model_present(mf),
         "verified": mf.verified}
        for mf in COMFY_MODELS + VAE_MANIFEST
    ]
    missing = [m["name"] for m in comfy_models if not m["present"]]
    sr_ckpt = _scale_rae_ckpt_present()
    sr_venv = config.PID_VENV_PYTHON.exists()
    return {
        "hfTokenPresent": bool(config.HF_TOKEN),
        "comfyInstalled": (config.COMFY_ROOT / "main.py").is_file(),
        "comfyRunning": comfy_up,
        "pidUpstreamCloned": (config.PID_REPO_DIR / "pid").is_dir(),
        "models": comfy_models,
        "modelsMissing": missing,
        "scaleRae": {"checkpoint": sr_ckpt, "venv": sr_venv, "ready": sr_ckpt and sr_venv},
        # "ready" reflects the default (flux/flux2) path; scale_rae is opt-in.
        "ready": comfy_up and not missing,
        "autoInstall": config.AUTO_INSTALL,
        "config": config.summary(),
    }


# ── Subprocess helpers ───────────────────────────────────────────────────────
async def _run_cmd(bus, step: str, args: list[str], cwd: Optional[Path] = None) -> int:
    _emit(bus, step, "running", " ".join(args[:4]) + " …")
    proc = await asyncio.create_subprocess_exec(
        *args, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    async for line in proc.stdout:  # type: ignore
        text = line.decode(errors="replace").rstrip()
        if text:
            _emit(bus, step, "log", text)
    return await proc.wait()


# ── Steps ────────────────────────────────────────────────────────────────────
async def ensure_comfyui(bus) -> None:
    if (config.COMFY_ROOT / "main.py").is_file():
        _emit(bus, "comfyui", "ok", f"ComfyUI present at {config.COMFY_ROOT}", 1.0)
        return
    if not shutil.which("git"):
        _emit(bus, "comfyui", "error", "git not found on PATH; install git first.")
        raise RuntimeError("git not found")
    config.COMFY_ROOT.parent.mkdir(parents=True, exist_ok=True)
    rc = await _run_cmd(bus, "comfyui", ["git", "clone", "--depth", "1", COMFYUI_GIT, str(config.COMFY_ROOT)])
    if rc != 0:
        _emit(bus, "comfyui", "error", "git clone failed")
        raise RuntimeError("ComfyUI clone failed")
    _emit(bus, "comfyui", "ok",
          "ComfyUI cloned. Install its Python deps + a CUDA torch build before "
          "first run (see docs/INSTALL.md).", 1.0)


async def ensure_pid_upstream(bus) -> None:
    if (config.PID_REPO_DIR / "pid").is_dir():
        _emit(bus, "pid-upstream", "ok", "pid-upstream present", 1.0)
        return
    if not shutil.which("git"):
        _emit(bus, "pid-upstream", "skip", "git missing; scale_rae backbone unavailable")
        return
    config.PID_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    rc = await _run_cmd(bus, "pid-upstream", ["git", "clone", "--depth", "1", PID_UPSTREAM_GIT, str(config.PID_REPO_DIR)])
    _emit(bus, "pid-upstream", "ok" if rc == 0 else "skip",
          "cloned" if rc == 0 else "clone failed (scale_rae optional)", 1.0)


def _link_or_copy(cached: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(cached, dest)          # hardlink: no extra disk
    except OSError:
        shutil.copy2(cached, dest)     # cross-volume fallback


def _hf_download(mf: ModelFile, token: Optional[str]) -> Path:
    """Download one file with huggingface_hub straight into the dest path."""
    from huggingface_hub import hf_hub_download

    dest = config.COMFY_MODELS_ROOT / mf.dest_subdir / mf.dest_name
    cached = hf_hub_download(repo_id=mf.repo, filename=mf.path, token=token or None)
    _link_or_copy(cached, dest)
    return dest


def _hf_download_to(root: Path, repo: str, path: str, token: Optional[str], dest_rel: str) -> Path:
    """Download a file and mirror it under ``root/dest_rel`` (used for the
    nvidia/PiD .pth checkpoints that live outside the ComfyUI models tree)."""
    from huggingface_hub import hf_hub_download

    dest = root / dest_rel
    if dest.exists():
        return dest
    cached = hf_hub_download(repo_id=repo, filename=path, token=token or None)
    _link_or_copy(cached, dest)
    return dest


async def download_models(bus) -> None:
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    pending = [mf for mf in COMFY_MODELS if not _model_present(mf)]
    total = len(pending)
    if total == 0:
        _emit(bus, "models", "ok", "all ComfyUI PiD models present", 1.0)
    for i, mf in enumerate(pending):
        _emit(bus, "models", "running", f"downloading {mf.dest_name}", i / max(1, total))
        try:
            await asyncio.to_thread(_hf_download, mf, token)
            _emit(bus, "models", "log", f"✓ {mf.dest_name}", (i + 1) / max(1, total))
        except Exception as e:
            _emit(bus, "models", "error",
                  f"failed {mf.dest_name}: {e}. Did you set HF_TOKEN and accept the "
                  f"model license on huggingface.co/{mf.repo}?")
            raise

    # Encode VAEs (public Comfy/Kijai repackages). Non-fatal: a failure here only
    # affects the flux/flux2 encode step, and the source can be overridden.
    for mf in VAE_MANIFEST:
        if _model_present(mf):
            continue
        if not (mf.repo and mf.path):
            _emit(bus, "models", "warn",
                  f"VAE {mf.dest_name} has no source; set PID_FLUX_VAE_REPO/PATH. See docs/INSTALL.md.")
            continue
        try:
            _emit(bus, "models", "running", f"downloading {mf.dest_name} (from {mf.repo})")
            await asyncio.to_thread(_hf_download, mf, token)
            _emit(bus, "models", "log", f"✓ {mf.dest_name}")
        except Exception as e:
            _emit(bus, "models", "warn", f"VAE {mf.dest_name} download failed: {e}")
    _emit(bus, "models", "ok", "model download step complete", 1.0)


# ── Scale-RAE (8×) subprocess backbone — full provisioning ───────────────────
def _scale_rae_ckpt_present() -> bool:
    return (config.PID_MODELS_ROOT / SCALE_RAE_CKPT_REL).is_file()


async def ensure_scale_rae(bus) -> None:
    """Provision the Scale-RAE 8× subprocess backbone end to end:
      1. clone nv-tlabs/PiD (reuses ensure_pid_upstream)
      2. build a dedicated venv at PID_HOME/pid-venv (config.PID_PYTHON auto-uses it)
      3. install PyTorch (CUDA) + the PiD subprocess deps + editable pid-upstream
      4. download the gated nvidia/PiD sr8x checkpoint
    """
    token = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        _emit(bus, "scale-rae", "error", "HF_TOKEN required for the nvidia/PiD checkpoint.")
        raise RuntimeError("HF_TOKEN missing")

    # 1) source repo
    await ensure_pid_upstream(bus)
    if not (config.PID_REPO_DIR / "pid").is_dir():
        _emit(bus, "scale-rae", "error", "pid-upstream not available (git clone failed).")
        raise RuntimeError("pid-upstream missing")

    # 2) dedicated venv
    if not config.PID_VENV_PYTHON.exists():
        _emit(bus, "scale-rae", "running", f"creating venv at {config.PID_VENV_DIR}")
        rc = await _run_cmd(bus, "scale-rae", [sys.executable, "-m", "venv", str(config.PID_VENV_DIR)])
        if rc != 0:
            _emit(bus, "scale-rae", "error", "venv creation failed")
            raise RuntimeError("pid-venv creation failed")
    py = str(config.PID_VENV_PYTHON)

    # 3) deps: pip, torch (CUDA), subprocess reqs, editable pid-upstream
    await _run_cmd(bus, "scale-rae", [py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    _emit(bus, "scale-rae", "running", f"installing PyTorch ({config.PID_TORCH_INDEX_URL}) — large, be patient")
    rc = await _run_cmd(bus, "scale-rae",
                        [py, "-m", "pip", "install", "torch", "torchvision",
                         "--index-url", config.PID_TORCH_INDEX_URL])
    if rc != 0:
        _emit(bus, "scale-rae", "warn", "torch install returned nonzero (continuing; check CUDA index).")
    reqs = Path(__file__).resolve().parent / "pid_subprocess_requirements.txt"
    rc = await _run_cmd(bus, "scale-rae", [py, "-m", "pip", "install", "-r", str(reqs)])
    if rc != 0:
        _emit(bus, "scale-rae", "error", "PiD subprocess requirements failed")
        raise RuntimeError("pid subprocess deps failed")
    rc = await _run_cmd(bus, "scale-rae", [py, "-m", "pip", "install", "-e", str(config.PID_REPO_DIR)])
    if rc != 0:
        _emit(bus, "scale-rae", "warn",
              "`pip install -e pid-upstream` failed; the PYTHONPATH fallback in server.py still works.")

    # 4) gated checkpoint
    if not _scale_rae_ckpt_present():
        _emit(bus, "scale-rae", "running", "downloading sr8x checkpoint (nvidia/PiD, gated)")
        try:
            await asyncio.to_thread(
                _hf_download_to, config.PID_MODELS_ROOT, SCALE_RAE_REPO,
                SCALE_RAE_CKPT_REL, token, SCALE_RAE_CKPT_REL,
            )
        except Exception as e:
            _emit(bus, "scale-rae", "error",
                  f"checkpoint download failed: {e}. Accept the license at "
                  f"huggingface.co/{SCALE_RAE_REPO} and check HF_TOKEN.")
            raise
    _emit(bus, "scale-rae", "ok", "Scale-RAE 8× ready", 1.0)


# ── Orchestrator ─────────────────────────────────────────────────────────────
# Scale-RAE is opt-in (it pulls PyTorch + a gated multi-GB checkpoint); the
# default one-click install covers the ComfyUI flux/flux2 path.
ALL_STEPS = ["comfyui", "models", "pid-upstream"]


async def run(bus, comfy, steps: Optional[list[str]] = None) -> dict:
    steps = steps or ALL_STEPS
    _emit(bus, "all", "running", f"starting setup: {', '.join(steps)}")
    try:
        if "comfyui" in steps:
            await ensure_comfyui(bus)
        if "models" in steps:
            if not config.HF_TOKEN and not os.environ.get("HF_TOKEN"):
                _emit(bus, "models", "error",
                      "HF_TOKEN is not set. Put it in .env (see .env.example) and restart.")
                raise RuntimeError("HF_TOKEN missing")
            await download_models(bus)
        if "pid-upstream" in steps:
            await ensure_pid_upstream(bus)
        if "scale-rae" in steps:
            await ensure_scale_rae(bus)
        _emit(bus, "all", "done", "setup complete", 1.0)
        return await status(comfy)
    except Exception as e:
        _emit(bus, "all", "error", str(e))
        return await status(comfy)
