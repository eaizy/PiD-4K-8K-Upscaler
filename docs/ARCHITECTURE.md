# Architecture

PiD Studio is a thin, decoupled web app around NVIDIA's PiD super-resolution.
It was extracted from a larger Tauri desktop app; the Tauri shell was dropped in
favour of a plain browser + FastAPI split so a fork needs no Rust toolchain.

```
┌────────────────────────────────────────────────────────────────────┐
│ Browser  —  Vite + React + shadcn/ui (playground layout)             │
│   • parameter panel · compare slider · live preview · setup wizard   │
└───────────────┬───────────────────────────────┬──────────────────────┘
   HTTP /upscale, /projects, /setup        WS /preview (progress + xt-step)
                │                               │
┌───────────────▼───────────────────────────────▼──────────────────────┐
│ FastAPI backend (backend/server.py)                                    │
│   • single-job queue + EventBus → WebSocket broadcast                  │
│   • project/file storage (projects.py) — replaces Tauri IPC            │
│   • auto-installer (installer.py) — ComfyUI + HF model download        │
└───────┬───────────────────────────────────────────────┬───────────────┘
        │ flux / flux2                                    │ scale_rae (8×)
┌───────▼──────────────────────────────┐     ┌────────────▼──────────────┐
│ ComfyUI (:8188) native PiD node graph │     │ PiD reference CLI          │
│  LoadImage→VAEEncode→PiDConditioning  │     │  python -m pid._src.       │
│  →SamplerCustom(lcm,manual sigmas)    │     │  inference.from_clean_siglip│
│  →VAEDecode(pixel_space)→SaveImage     │     │  (subprocess)              │
└───────────────────────────────────────┘     └────────────────────────────┘
```

## Why this split

The original sidecar already spoke HTTP/WS; Tauri only spawned it and did file
IPC. Removing Tauri means:

- **Reused as-is:** the ComfyUI orchestration (`comfy_workflow.py`,
  `comfy_client.py`) — the node graph, the PiD distill sigma schedules, the
  dual-pass 8K strategy, VRAM guards, and model-readiness detection. This is the
  hard-won, empirically-tuned part and is the most expensive thing to rebuild.
- **Rewritten:** the Tauri file IPC → HTTP endpoints in `projects.py` + `server.py`.
- **Dropped:** the Tauri/Rust shell and the original bespoke UI (replaced by the
  shadcn playground layout).

## Backend endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model + ComfyUI readiness flags |
| POST | `/upscale` | enqueue a job → `{jobId, queuePosition}` |
| WS | `/preview` | `progress` / `xt-step` / `done` / `error` / `setup` events |
| GET | `/models` | local checkpoint status |
| GET/POST | `/projects` | list / create |
| GET/PUT/DELETE | `/projects/{slug}` | manifest read / write / delete |
| POST | `/projects/{slug}/images` | upload (multipart) |
| GET/DELETE | `/projects/{slug}/images/{key}` | serve / delete image |
| GET | `/setup/status` | what's installed vs missing |
| POST | `/setup/run` | start auto-install (streams `setup` events) |

## Backbones & models

| Backbone | Path | Scale | Runs in |
|---|---|---|---|
| `flux` | ComfyUI native PiD graph | 4× (16-ch VAE) | ComfyUI |
| `flux2` | ComfyUI native PiD graph | 4× (128-ch VAE) | ComfyUI |
| `scale_rae` | `from_clean_siglip` CLI | 8× single-pass | subprocess |

Quality → input long edge → output: `2k` 512→2048, `4k` 1024→4096,
`8k` = 4K PiD + Lanczos 2× (true 7680² pixel-diffusion is infeasible on consumer
VRAM). Distilled steps: `1` (fast) or `4` (quality); sigma schedule
`0.999,0.866,0.634,0.342,0` for 4-step.

### ComfyUI model files (downloaded by the installer)

| File | Dir | Source |
|---|---|---|
| `pid_flux1_512_to_2048_4step_bf16.safetensors` | `diffusion_models/` | Comfy-Org/PixelDiT |
| `pid_flux1_1024_to_4096_4step_bf16.safetensors` | `diffusion_models/` | Comfy-Org/PixelDiT |
| `pid_flux2_512_to_2048_4step_bf16.safetensors` | `diffusion_models/` | Comfy-Org/PixelDiT |
| `pid_flux2_1024_to_4096_4step_bf16.safetensors` | `diffusion_models/` | Comfy-Org/PixelDiT |
| `gemma_2_2b_it_elm_bf16.safetensors` | `text_encoders/` | Comfy-Org/PixelDiT |
| `flux-vae-bf16.safetensors` | `vae/` | Kijai/flux-fp8 |
| `flux2-vae.safetensors` | `vae/` | Comfy-Org/flux2-dev (`split_files/vae/`) |

The `scale_rae` subprocess uses the reference `.pth` checkpoints from
`nvidia/PiD` (e.g. `checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth`).

## Data layout

```
data/<slug>/
  manifest.json          # title, status, inputKey, outputs[], params
  input.{png,jpg,webp}
  output_<epoch>.png
  job_<jobId>/           # per-job working dir (preprocess, comfy out, 8k)
```

## Configuration

All paths and ports resolve in `backend/config.py` from environment variables
(see `.env.example`), defaulting to self-contained locations under the repo
(`.pidstudio/`, `data/`). There are no hard-coded absolute paths.

## Licensing boundary

The app **orchestrates** but never redistributes third-party weights/code: PiD
code is Apache-2.0 (cloned at runtime), PiD weights are NSCLv1 (downloaded with
the user's token), ComfyUI is GPL-3.0 (cloned at runtime). The app's own code is
Apache-2.0. See [`NOTICE`](../NOTICE).
