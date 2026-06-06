# PiD Studio

A standalone web app for **NVIDIA PiD** ([Pixel Diffusion](https://github.com/nv-tlabs/PiD)) super-resolution. Drop in an image, pick a target quality, and watch it upscale 4K/8K in your browser — with a live diffusion preview.

> Built on a [shadcn/ui playground](https://ui.shadcn.com/examples/playground)-style interface. Flux / Flux.2 backbones run through a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend; Scale-RAE (8×) runs via the PiD reference CLI.

---

## ⚠️ Licensing (read first)

| Component | License | Notes |
|---|---|---|
| **This app's code** | Apache-2.0 | Free to use, modify, redistribute. |
| **PiD source code** (`nv-tlabs/PiD`) | Apache-2.0 | Cloned at runtime. |
| **PiD model weights** (`nvidia/PiD`) | **NSCLv1 — non-commercial** | **Research / evaluation only.** Downloaded at runtime with *your* `HF_TOKEN`; never committed here. |
| **ComfyUI** | GPL-3.0 | Optionally auto-installed at runtime. |

By downloading and running the PiD weights you agree to NVIDIA's NSCLv1 terms. See [`NOTICE`](NOTICE).

---

## Quick start (minimal install)

```bash
git clone <this-repo> pid-studio && cd pid-studio
cp .env.example .env          # then put your HF_TOKEN in .env
python scripts/bootstrap.py   # installs deps, ComfyUI, and downloads models
python scripts/run.py         # starts backend + frontend, opens the browser
```

That's it. On first launch the app detects what's missing (ComfyUI, weights)
and downloads everything into the right place using your `HF_TOKEN`. You only
ever provide that one token.

> **GPU:** A CUDA GPU with ~16 GB VRAM is recommended for 4K. See [docs/INSTALL.md](docs/INSTALL.md).

---

## Architecture

```
Browser (Vite + React + shadcn)
   │  HTTP /upscale, /projects…   WS /preview (live xt-step + progress)
   ▼
FastAPI backend  ──subprocess──▶  PiD reference CLI   (scale_rae 8×)
   │  HTTP /prompt, WS /ws
   ▼
ComfyUI  ──▶  native PiD node graph (flux / flux2 → 2K/4K, 8K = 4K+Lanczos)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown.

## Features

- **One-token setup** — provide `HF_TOKEN`; the app installs ComfyUI and
  downloads the right model files into the right folders on first run.
- **Live diffusion preview** — watch the latent decode over the WebSocket as it
  upscales.
- **Before/after compare slider** and a per-project gallery.
- **Three backbones** — Flux & Flux.2 (4×, via ComfyUI) and Scale-RAE (8×, via
  the PiD reference CLI).
- **VRAM-safe** — output-pixel budgeting + NaN/black detection so a too-large
  target fails loud instead of hanging.

## Repository layout

| Path | What |
|---|---|
| `backend/` | FastAPI server, ComfyUI client, PiD workflow, auto-installer |
| `frontend/` | Vite + React + shadcn playground UI |
| `scripts/` | `bootstrap.py` (setup), `run.py` (launch) |
| `docs/` | [Install](docs/INSTALL.md) + [Architecture](docs/ARCHITECTURE.md) |

## Development

```bash
python scripts/bootstrap.py          # deps
python scripts/run.py                # backend :17820 + frontend :5173

# tests run against a live backend; GPU paths auto-skip without ComfyUI/weights
.venv/Scripts/python -m pytest backend
```

The frontend talks to the backend through a `/api` proxy (see
[`vite.config.ts`](frontend/vite.config.ts)); no CORS setup needed in dev.

## Roadmap

- [ ] **App screenshots & demo media** — add UI screenshots and a short capture of
  the playground (compare slider + live preview) to this README.
- [ ] **Video upscaling** — frame-by-frame PiD upscaling for video input, with
  temporal consistency and audio passthrough.

## Status

Extracted from a larger application as a standalone, public research tool. The
backend port is verified end-to-end (project + file API tested live; the ComfyUI
upscale path mirrors the original sidecar's e2e tests).
