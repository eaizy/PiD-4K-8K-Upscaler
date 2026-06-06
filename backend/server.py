"""
PiD Studio backend — FastAPI server.

Architecture
------------
Browser → HTTP(POST /upscale)  → Backend → ComfyUI (flux/flux2 native PiD graph)
        ↘ WebSocket(/preview)   ←  push progress + live latent2rgb preview frames
                                 ↘ subprocess(PiD CLI)  (scale_rae 8×)

This is a de-coupled port of the Uniqlaw Tauri sidecar. The Tauri shell is gone:
the browser talks to this server directly over HTTP/WS, and the project/file
operations that Tauri used to handle via IPC are now plain HTTP endpoints
(see projects.py). All paths come from config.py (env-driven, cross-platform).

A single job runs at a time — concurrent upscales are queued (pixel-space
diffusion VRAM scales with output pixels; 2 concurrent risks OOM).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

import config
import projects
from comfy_client import ComfyClient
from comfy_workflow import build_workflow, model_file

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pid-studio")

comfy = ComfyClient(host=config.COMFY_HOST, port=config.COMFY_PORT)


# ── Checkpoint locations (subprocess scale_rae path) ────────────────────────
def _ckpt(name: str) -> Path:
    return config.PID_MODELS_ROOT / "checkpoints" / name / "model_ema_bf16.pth"


CHECKPOINT_2K = _ckpt("PiD_res2k_sr4x_official_flux_distill_4step")
CHECKPOINT_4K = _ckpt("PiD_res2kto4k_sr4x_official_flux_distill_4step")
CHECKPOINT_FLUX2_2K = _ckpt("PiD_res2k_sr4x_official_flux2_distill_4step")
CHECKPOINT_FLUX2_4K = _ckpt("PiD_res2kto4k_sr4x_official_flux2_distill_4step")
CHECKPOINT_SCALE_RAE = _ckpt("PiD_res2k_sr8x_official_siglip_distill_4step")


def comfy_model_ready(backbone: str, quality: str) -> bool:
    """Is the PiD safetensors present under ComfyUI diffusion_models/ for this
    backbone? 8k → uses the 4k model (4k PiD + Lanczos 2×)."""
    q = "2k" if quality == "2k" else "4k"
    try:
        fn = model_file(backbone, q)
    except KeyError:
        return False
    return (config.COMFY_MODELS_ROOT / "diffusion_models" / fn).is_file()


# ── Request / event schemas ─────────────────────────────────────────────────
class UpscaleRequest(BaseModel):
    slug: str = Field(..., description="PiD project slug")
    inputKey: str = Field(..., description="Input file key (no extension)")
    # flux       → ComfyUI, 4× (16-ch VAE),  2k/4k/8k(=4k+Lanczos)
    # flux2      → ComfyUI, 4× (128-ch VAE), 2k/4k/8k(=4k+Lanczos)
    # scale_rae  → subprocess from_clean_siglip, 8× single-pass, 256→2048
    backbone: str = Field("flux", pattern="^(flux|flux2|scale_rae)$")
    targetQuality: str = Field("4k", pattern="^(2k|4k|8k)$")
    # PiD DMD2-distilled: meaningful values are 1 (fast, single-shot) or 4
    # (quality). Above 4 is a no-op. UI exposes a Fast/Quality toggle (1|4).
    inferenceSteps: int = Field(4, ge=1, le=4)
    degradeSigma: float = Field(0.0, ge=0.0, le=1.0)
    saveIntermediate: bool = True
    seed: int = Field(5, ge=0)
    prompt: Optional[str] = Field(None)


class HealthResponse(BaseModel):
    status: str
    models2kReady: bool
    models4kReady: bool
    fluxVaeReady: bool
    flux2Ready: bool
    scaleRaeReady: bool
    pidRepoReady: bool
    comfyReady: bool = False
    comfyFlux2kReady: bool = False
    comfyFlux4kReady: bool = False
    comfyFlux22kReady: bool = False
    comfyFlux24kReady: bool = False
    jobsInQueue: int
    currentJobId: Optional[str]


class UpscaleResponse(BaseModel):
    jobId: str
    queuePosition: int


class CreateProjectRequest(BaseModel):
    title: str = ""
    slug: Optional[str] = None


# ── Job & WebSocket bus ──────────────────────────────────────────────────────
@dataclass
class Job:
    job_id: str
    slug: str
    request: UpscaleRequest
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    status: str = "queued"
    error: Optional[str] = None
    output_dir: Optional[Path] = None


class EventBus:
    """Simple in-process pub/sub. Broadcast to every subscribed WS client."""

    def __init__(self) -> None:
        self._subs: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def publish(self, event: dict) -> None:
        for q in self._subs:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except Exception:
                pass


bus = EventBus()
jobs_queue: asyncio.Queue[Job] = asyncio.Queue()
current_job: Optional[Job] = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def png_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def pick_checkpoint(target: str) -> Path:
    return CHECKPOINT_4K if target == "4k" else CHECKPOINT_2K


# Long edge (px) the input is downscaled to before PiD's fixed 4× upscale:
#   2k → 512  → 2048 ; 4k → 1024 → 4096 ; 8k → 480 (dual-pass base)
PID_TARGET_LONG_EDGE = {"2k": 512, "4k": 1024, "8k": 480}

# VRAM safety budget: PiD pixel-space diffusion VRAM scales with OUTPUT pixels
# (denoising directly on H×W×3, not latents). Proven ceiling on a 24 GB card:
# 9.4 MP worked, 16.7 MP overflowed → black (bf16 NaN). Cap at 10.5 MP; presets
# exceeding it (square inputs especially) shrink the input proportionally.
PID_MAX_OUTPUT_PIXELS = 10_500_000


def preprocess_input(
    src: Path,
    target: str,
    work_dir: Path,
    scale: int = 4,
    long_edge_override: Optional[int] = None,
    out_name: str = "_resized_input.png",
) -> Path:
    """Aspect-preserving downscale to the target long edge + VRAM-safe output
    budget. Edges rounded to a multiple of 16 (Flux VAE latent grid)."""
    from PIL import Image

    long_edge = long_edge_override or PID_TARGET_LONG_EDGE.get(target, 1024)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = long_edge / max(w, h)
    nw, nh = (w, h) if s >= 1.0 else (round(w * s), round(h * s))

    out_px = (nw * scale) * (nh * scale)
    if out_px > PID_MAX_OUTPUT_PIXELS:
        shrink = math.sqrt(PID_MAX_OUTPUT_PIXELS / out_px)
        nw, nh = round(nw * shrink), round(nh * shrink)
        log.info(
            "VRAM budget: output %.1f MP > %.1f MP cap → input shrunk by %.2f",
            out_px / 1e6, PID_MAX_OUTPUT_PIXELS / 1e6, shrink,
        )

    nw = max(16, (nw // 16) * 16)
    nh = max(16, (nh // 16) * 16)
    if (nw, nh) == (w, h):
        return src
    resized = img.resize((nw, nh), Image.LANCZOS)
    out = work_dir / out_name
    resized.save(out)
    log.info("Preprocess: %dx%d → %dx%d (target=%s)", w, h, nw, nh, target)
    return out


PID_DEFAULT_PROMPT = "a high quality photo, sharp focus, detailed"

BACKBONE_CONFIG = {
    "flux": {
        "module": "pid._src.inference.from_clean_flux",
        "scale": 4,
        "keep_input_size": True,
        "input_resolution": None,
    },
    "flux2": {
        "module": "pid._src.inference.from_clean_flux2",
        "scale": 4,
        "keep_input_size": True,
        "input_resolution": None,
    },
    "scale_rae": {
        "module": "pid._src.inference.from_clean_siglip",
        "scale": 8,
        "keep_input_size": False,
        "input_resolution": 256,
    },
}


def build_pass_args(
    input_path: Path,
    output_dir: Path,
    backbone: str,
    ckpt_path: Path,
    ckpt_type: str,
    inference_steps: int,
    degrade_sigma: float,
    seed: int = 5,
    prompt: Optional[str] = None,
) -> list[str]:
    """CLI args for one PiD subprocess pass (backbone-aware). Flag names are
    LOCKED to the PiD upstream --help."""
    cfg = BACKBONE_CONFIG.get(backbone, BACKBONE_CONFIG["flux"])
    effective_prompt = (prompt or "").strip() or PID_DEFAULT_PROMPT
    args = [
        config.PID_PYTHON,
        "-m",
        cfg["module"],
        "--input_path", str(input_path),
        "--output_dir", str(output_dir),
        "--checkpoint_path", str(ckpt_path),
        "--pid_ckpt_type", ckpt_type,
        "--pid_inference_steps", str(inference_steps),
        "--scale", str(cfg["scale"]),
        "--degrade_sigmas", f"{degrade_sigma:.3f}",
        "--seed", str(seed),
        "--save_format", "png",
        "--prompt", effective_prompt,
    ]
    if cfg["keep_input_size"]:
        args.append("--keep_input_size")
    else:
        args.extend(["--input_resolution", str(cfg["input_resolution"])])
    return args


def find_pass_output(output_dir: Path) -> Optional[Path]:
    """Find the real upscaled PNG in a PiD pass output_dir. PiD writes
    flux_PiD_*/sigma_*/<basename>.png (real 4×) + vae_decode/... (baseline)."""
    all_pngs = list(output_dir.rglob("*.png"))
    if not all_pngs:
        return None
    pid_outputs = [p for p in all_pngs if "flux_PiD" in str(p) or "PiD_res" in str(p)]
    candidates = sorted(
        pid_outputs if pid_outputs else all_pngs,
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return candidates[0]


_PASS_MILESTONES = [
    ("Loading our pixel decoder", 0.10, "model-load"),
    ("load_text_encoder", 0.40, "model-load"),
    ("Time spent on instantiate model", 0.55, "model-load"),
    ("Loading model from consolidated", 0.65, "model-load"),
    ("Processing", 0.72, "encoding"),
    ("Clean latent shape", 0.78, "encoding"),
    ("[distill inference]", 0.82, "denoising"),
    ("sigma=", 0.95, "saving"),
    ("Done!", 0.99, "saving"),
]


async def run_pid_pass(
    job: Job,
    input_path: Path,
    work_dir: Path,
    backbone: str,
    ckpt_path: Path,
    ckpt_type: str,
    progress_base: float,
    progress_span: float,
    phase_prefix: str = "",
) -> Path:
    """Run a single PiD subprocess pass, return the output PNG path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    args = build_pass_args(
        input_path, work_dir, backbone, ckpt_path, ckpt_type,
        job.request.inferenceSteps, job.request.degradeSigma,
        seed=job.request.seed, prompt=job.request.prompt,
    )
    log.info("Spawning PiD pass (%s/%s): %s", backbone, phase_prefix or ckpt_type, " ".join(args))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.PID_REPO_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    cwd = str(config.PID_REPO_DIR) if config.PID_REPO_DIR.exists() else str(config.PID_MODELS_ROOT)

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    nan_detected = False
    async for line in proc.stdout:  # type: ignore
        decoded = line.decode(errors="replace").rstrip()
        if not decoded:
            continue
        log.info("[pid] %s", decoded)
        if "invalid value encountered in cast" in decoded:
            nan_detected = True
        for needle, raw_prog, phase in _PASS_MILESTONES:
            if needle in decoded:
                mapped = progress_base + raw_prog * progress_span
                bus.publish({
                    "type": "progress",
                    "slug": job.slug,
                    "jobId": job.job_id,
                    "phase": f"{phase_prefix}{phase}",
                    "progress01": round(mapped, 4),
                })
                break

    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"PiD pass exited with code {rc}")
    if nan_detected:
        raise RuntimeError(
            "PiD output contained NaN/Inf (black image) — likely VRAM exhaustion "
            "at this resolution. Try a lower target quality or a smaller input."
        )
    out = find_pass_output(work_dir)
    if out is None:
        raise RuntimeError(f"PiD pass produced no output PNG in {work_dir}")
    return out


# ── ComfyUI-backed upscale (flux / flux2) — native PiD node graph ────────────
COMFY_INPUT_LONG_EDGE = {"2k": 512, "4k": 1024, "8k": 1024}


def preprocess_for_comfy(
    src: Path, target_quality: str, work_dir: Path
) -> tuple[Path, int, int]:
    """Downscale the input to the backbone's expected scale, return (path, w, h).
    Aspect-preserving, edges a multiple of 16 (Flux VAE latent grid)."""
    from PIL import Image

    long_edge = COMFY_INPUT_LONG_EDGE.get(target_quality, 1024)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    s = long_edge / max(w, h)
    nw, nh = (w, h) if s >= 1.0 else (round(w * s), round(h * s))
    nw = max(16, (nw // 16) * 16)
    nh = max(16, (nh // 16) * 16)
    if (nw, nh) == (w, h) and src.suffix.lower() == ".png":
        return src, w, h
    resized = img.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else img
    out = work_dir / "_comfy_input.png"
    resized.save(out)
    log.info("Comfy preprocess: %dx%d → %dx%d (out=%dx%d)", w, h, nw, nh, nw * 4, nh * 4)
    return out, nw, nh


async def run_comfy_upscale(job: Job, input_path: Path, output_dir: Path) -> Path:
    """Run flux/flux2 through the ComfyUI native PiD graph; return the final PNG.
    8K = 4K PiD + Lanczos 2× (true 7680² pixel-diffusion is infeasible here)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    req = job.request
    backbone = req.backbone
    quality = req.targetQuality
    pid_quality = "4k" if quality in ("4k", "8k") else "2k"

    if not await comfy.health():
        raise RuntimeError(
            f"ComfyUI ({config.COMFY_HOST}:{config.COMFY_PORT}) is not responding. "
            "Is ComfyUI running?"
        )

    # 1) preprocess + upload
    proc_in, in_w, in_h = preprocess_for_comfy(input_path, pid_quality, output_dir)
    out_w, out_h = in_w * 4, in_h * 4
    bus.publish({"type": "progress", "slug": job.slug, "jobId": job.job_id,
                 "phase": "encoding", "progress01": 0.08})
    comfy_ref = await comfy.upload_image(proc_in)
    log.info("Comfy upload: %s → %s (out %dx%d, model=%s)",
             proc_in.name, comfy_ref, out_w, out_h, model_file(backbone, pid_quality))

    # 2) workflow build + submit
    workflow = build_workflow(
        image_filename=comfy_ref, backbone=backbone, target_quality=pid_quality,
        out_w=out_w, out_h=out_h, seed=req.seed, prompt=req.prompt,
        degrade_sigma=req.degradeSigma, filename_prefix=f"pidstudio_{job.job_id}",
        inference_steps=req.inferenceSteps,
    )
    client_id = f"pidstudio_{job.job_id}"
    prompt_id = await comfy.submit_prompt(workflow, client_id)
    log.info("Comfy prompt submitted: %s", prompt_id)

    # 3) WS stream (real step progress + live latent2rgb preview)
    prog_cap = 0.92 if quality == "8k" else 0.99

    def on_prog(p: float, stage: str) -> None:
        phase = {"running": "denoising", "saving": "saving"}.get(stage, "denoising")
        bus.publish({"type": "progress", "slug": job.slug, "jobId": job.job_id,
                     "phase": phase, "progress01": round(0.08 + p * (prog_cap - 0.08), 4)})

    preview_count = {"n": 0}

    def on_preview(img_bytes: bytes) -> None:
        try:
            import io as _io
            from PIL import Image as _Img

            im = _Img.open(_io.BytesIO(img_bytes)).convert("RGB")
            buf = _io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return
        preview_count["n"] += 1
        bus.publish({
            "type": "xt-step", "slug": job.slug, "jobId": job.job_id,
            "step": preview_count["n"], "totalSteps": max(1, req.inferenceSteps),
            "imageBase64": b64, "filename": f"preview_{preview_count['n']}.png",
        })

    entry = await comfy.stream_execution(
        prompt_id, client_id, on_progress=on_prog, on_preview=on_preview, timeout_s=600,
    )
    meta = comfy.extract_output(entry)
    if not meta:
        raise RuntimeError(
            f"ComfyUI produced no output image. outputs={list(entry.get('outputs', {}).keys())}"
        )
    png_bytes = await comfy.fetch_image(meta)
    pid_out = output_dir / "_comfy_pid_out.png"
    pid_out.write_bytes(png_bytes)

    # NaN/black guard.
    from PIL import Image
    import numpy as np

    arr = np.asarray(Image.open(pid_out).convert("RGB"))
    if float(arr.std()) < 1.0:
        raise RuntimeError(
            "ComfyUI PiD output is flat/black (suspected NaN) — VRAM may be "
            "insufficient at this resolution. Try a lower target quality."
        )

    # 4) 8K post-upscale (Lanczos 2×).
    if quality == "8k":
        bus.publish({"type": "progress", "slug": job.slug, "jobId": job.job_id,
                     "phase": "saving", "progress01": 0.95})
        img = Image.open(pid_out).convert("RGB")
        w, h = img.size
        img8k = img.resize((w * 2, h * 2), Image.LANCZOS)
        final = output_dir / "_comfy_8k.png"
        img8k.save(final)
        log.info("8K post-upscale: %dx%d → %dx%d (Lanczos)", w, h, w * 2, h * 2)
        return final
    return pid_out


# ── Job runner ───────────────────────────────────────────────────────────────
async def run_job(job: Job) -> None:
    global current_job
    current_job = job
    job.status = "running"
    bus.publish({"type": "progress", "slug": job.slug, "jobId": job.job_id,
                 "phase": "encoding", "progress01": 0.02})

    project_dir = projects.project_dir(job.slug, create=True)
    output_dir = project_dir / f"job_{job.job_id}"
    output_dir.mkdir(exist_ok=True)
    job.output_dir = output_dir

    input_path = projects.find_image(job.slug, job.request.inputKey)
    if input_path is None:
        job.status = "error"
        job.error = f"Input not found for slug={job.slug} key={job.request.inputKey}"
        bus.publish({"type": "error", "slug": job.slug, "jobId": job.job_id, "message": job.error})
        current_job = None
        return

    try:
        backbone = job.request.backbone
        if backbone == "scale_rae":
            log.info("Scale-RAE 8x single-pass")
            final_path = await run_pid_pass(
                job, input_path, output_dir,
                backbone="scale_rae", ckpt_path=CHECKPOINT_SCALE_RAE, ckpt_type="2k",
                progress_base=0.0, progress_span=0.99, phase_prefix="",
            )
        else:
            log.info("%s → ComfyUI backend (%s)", backbone, job.request.targetQuality)
            final_path = await run_comfy_upscale(job, input_path, output_dir)

        # Save final output into the project dir.
        final_key = f"output_{int(time.time())}"
        target = project_dir / f"{final_key}.png"
        shutil.copy2(final_path, target)
        log.info("Output: %s → %s", final_path.name, target.name)
        job.finished_at = time.time()
        job.status = "done"

        # Record the output in the manifest so the gallery picks it up.
        dims = projects.image_dimensions(target)
        _append_output_to_manifest(job, final_key, dims)

        bus.publish({"type": "progress", "slug": job.slug, "jobId": job.job_id,
                     "phase": "saving", "progress01": 1.0})
        bus.publish({
            "type": "done", "slug": job.slug, "jobId": job.job_id,
            "outputKey": final_key, "outputPath": str(target),
            "durationMs": int((job.finished_at - job.started_at) * 1000),
        })
    except FileNotFoundError as e:
        job.status = "error"
        job.error = f"PiD executable / Python not found: {e}"
        bus.publish({"type": "error", "slug": job.slug, "jobId": job.job_id, "message": job.error})
    except RuntimeError as e:
        job.status = "error"
        job.error = str(e)
        bus.publish({"type": "error", "slug": job.slug, "jobId": job.job_id, "message": job.error})
    except Exception as e:
        job.status = "error"
        job.error = f"Unexpected: {e}"
        bus.publish({"type": "error", "slug": job.slug, "jobId": job.job_id, "message": job.error})
    finally:
        current_job = None


def _append_output_to_manifest(job: Job, key: str, dims: Optional[tuple[int, int]]) -> None:
    """Append the finished output to the project manifest (server-side so the
    gallery survives a page reload). Best-effort; never fails the job."""
    try:
        m = projects.load_manifest(job.slug)
        if not m:
            return
        entry = {
            "key": key,
            "kind": "output",
            "resolution": list(dims) if dims else None,
            "createdAt": int(time.time() * 1000),
            "absolutePath": str(projects.project_dir(job.slug) / f"{key}.png"),
            "params": {
                "backbone": job.request.backbone,
                "targetQuality": job.request.targetQuality,
                "inferenceSteps": job.request.inferenceSteps,
                "degradeSigma": job.request.degradeSigma,
                "seed": job.request.seed,
                "prompt": job.request.prompt,
                "sourceKey": job.request.inputKey,
            },
        }
        m["outputs"] = (m.get("outputs") or []) + [entry]
        m["status"] = "done"
        m["updatedAt"] = int(time.time() * 1000)
        projects.save_manifest(job.slug, m)
    except Exception as e:
        log.warning("manifest append failed: %s", e)


async def job_worker() -> None:
    while True:
        job = await jobs_queue.get()
        try:
            await run_job(job)
        except Exception as e:
            log.exception("job_worker crash: %s", e)
        finally:
            jobs_queue.task_done()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="PiD Studio Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_start() -> None:
    config.ensure_runtime_dirs()
    asyncio.create_task(job_worker())
    log.info("Backend ready. %s", json.dumps(config.summary()))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    comfy_up = await comfy.health()
    return HealthResponse(
        status="ready",
        models2kReady=CHECKPOINT_2K.is_file(),
        models4kReady=CHECKPOINT_4K.is_file(),
        fluxVaeReady=config.PID_FLUX_VAE.is_file(),
        flux2Ready=CHECKPOINT_FLUX2_2K.is_file()
        and CHECKPOINT_FLUX2_4K.is_file()
        and config.PID_FLUX2_VAE.is_file(),
        scaleRaeReady=CHECKPOINT_SCALE_RAE.is_file(),
        pidRepoReady=config.PID_REPO_DIR.is_dir() and (config.PID_REPO_DIR / "pid").is_dir(),
        comfyReady=comfy_up,
        comfyFlux2kReady=comfy_model_ready("flux", "2k"),
        comfyFlux4kReady=comfy_model_ready("flux", "4k"),
        comfyFlux22kReady=comfy_model_ready("flux2", "2k"),
        comfyFlux24kReady=comfy_model_ready("flux2", "4k"),
        jobsInQueue=jobs_queue.qsize(),
        currentJobId=current_job.job_id if current_job else None,
    )


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(req: UpscaleRequest) -> UpscaleResponse:
    if req.backbone == "scale_rae":
        if not CHECKPOINT_SCALE_RAE.is_file():
            raise HTTPException(503, detail="Scale-RAE (sr8x) checkpoint missing")
    else:
        if not comfy_model_ready(req.backbone, req.targetQuality):
            raise HTTPException(
                503,
                detail=f"ComfyUI PiD model missing: {req.backbone}/{req.targetQuality} "
                f"({config.COMFY_MODELS_ROOT / 'diffusion_models'})",
            )
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, slug=req.slug, request=req)
    await jobs_queue.put(job)
    bus.publish({"type": "progress", "slug": req.slug, "jobId": job_id,
                 "phase": "queued", "progress01": 0.0})
    return UpscaleResponse(jobId=job_id, queuePosition=jobs_queue.qsize())


# ── Project / file endpoints (replace the Tauri IPC commands) ────────────────
@app.get("/models")
async def models_status() -> dict:
    return projects.check_models()


@app.get("/projects")
async def list_projects() -> list[dict]:
    return projects.list_projects()


@app.post("/projects")
async def create_project(req: CreateProjectRequest) -> dict:
    return projects.create_project(req.title, req.slug)


@app.get("/projects/{slug}")
async def get_project(slug: str) -> JSONResponse:
    m = projects.load_manifest(slug)
    if m is None:
        raise HTTPException(404, detail="project not found")
    return JSONResponse(m)


@app.put("/projects/{slug}")
async def put_project(slug: str, manifest: dict) -> dict:
    manifest["updatedAt"] = int(time.time() * 1000)
    projects.save_manifest(slug, manifest)
    return manifest


@app.delete("/projects/{slug}")
async def delete_project(slug: str) -> dict:
    projects.delete_project(slug)
    return {"ok": True}


@app.post("/projects/{slug}/images")
async def upload_image(
    slug: str,
    key: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    raw = await file.read()
    p = projects.save_image_bytes(slug, key, raw, file.filename or "upload.png")
    dims = projects.image_dimensions(p)
    return {"key": key, "path": str(p), "resolution": list(dims) if dims else None}


@app.get("/projects/{slug}/images/{key}")
async def get_image(slug: str, key: str):
    p = projects.find_image(slug, key)
    if p is None:
        raise HTTPException(404, detail="image not found")
    return FileResponse(p, media_type=projects.image_mime(p))


@app.delete("/projects/{slug}/images/{key}")
async def remove_image(slug: str, key: str) -> dict:
    projects.delete_image(slug, key)
    return {"ok": True}


# ── Setup / installer endpoints ──────────────────────────────────────────────
@app.get("/setup/status")
async def setup_status() -> dict:
    """What's installed vs missing, so the frontend setup wizard can render."""
    import installer

    return await installer.status(comfy)


@app.post("/setup/run")
async def setup_run(steps: Optional[list[str]] = None) -> dict:
    """Kick off auto-install (ComfyUI + model download). Streams progress over
    the /preview WS as {type:'setup', ...} events; returns immediately."""
    import installer

    asyncio.create_task(installer.run(bus, comfy, steps))
    return {"started": True}


# ── WebSocket preview/progress stream ────────────────────────────────────────
@app.websocket("/preview")
async def preview_ws(ws: WebSocket) -> None:
    await ws.accept()
    q = bus.subscribe()
    log.info("WS client connected; subs=%d", len(bus._subs))
    try:
        while True:
            event = await q.get()
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("ws send failed: %s", e)
    finally:
        bus.unsubscribe(q)
        log.info("WS client disconnected; subs=%d", len(bus._subs))


def main() -> None:
    uvicorn.run("server:app", host="127.0.0.1", port=config.PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
