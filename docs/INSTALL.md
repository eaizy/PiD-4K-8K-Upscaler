# Installation

PiD Studio is designed so that a fork needs **one secret (`HF_TOKEN`)** and a
single bootstrap command; the app installs ComfyUI and downloads the model
weights itself on first run.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | for the backend + bootstrap |
| Node.js ≥ 18 | for the frontend |
| git | used to clone ComfyUI + the PiD repo at runtime |
| NVIDIA GPU | ~16 GB VRAM recommended for 4K; pixel-space diffusion scales with output pixels |
| Hugging Face account | with the `nvidia/PiD` license accepted (gated) |

## 1. Get a Hugging Face token

1. Visit <https://huggingface.co/nvidia/PiD> and **accept the license** (NSCLv1,
   non-commercial). The repo is gated; downloads fail until you accept.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. `cp .env.example .env` and set `HF_TOKEN=hf_...`.

## 2. Bootstrap (code deps only — fast)

```bash
python scripts/bootstrap.py
```

This creates `.venv`, installs `backend/requirements.txt`, installs the frontend
`node_modules`, and creates `.env` if missing. It does **not** download models.

## 3. Run

```bash
python scripts/run.py
```

Starts the FastAPI backend (`:17820`) and the Vite dev server (`:5173`) and opens
the browser. On first launch the **setup wizard** appears: click *Install
everything* and it will

1. clone ComfyUI into `.pidstudio/ComfyUI` (recent ComfyUI ships the PiD nodes
   in core — see [comfyanonymous/ComfyUI#14103](https://github.com/comfyanonymous/ComfyUI/pull/14103)),
2. download the PiD diffusion models + text encoder from
   [Comfy-Org/PixelDiT](https://huggingface.co/Comfy-Org/PixelDiT) into
   `ComfyUI/models/diffusion_models` and `text_encoders`,
3. clone `nv-tlabs/PiD` for the Scale-RAE (8×) subprocess backbone.

Progress streams live into the wizard.

## 4. ComfyUI Python environment

Cloning ComfyUI does not install its Python deps or a CUDA build of PyTorch.
Once ComfyUI is cloned, set it up in its own environment:

```bash
cd .pidstudio/ComfyUI
python -m venv venv
venv/Scripts/pip install -r requirements.txt          # bin/pip on macOS/Linux
venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124
python main.py                                          # starts ComfyUI on :8188
```

The backend talks to ComfyUI over HTTP/WS on `COMFY_PORT` (default 8188). When
ComfyUI is up, the *ComfyUI online* badge in the top bar turns green.

> Already have ComfyUI? Point `COMFY_ROOT` / `COMFY_PORT` at it in `.env` and skip
> the clone.

## 5. VAE files (auto-downloaded)

The Flux/Flux2 **encode** VAEs are pulled by the installer from public repos:

| File | Source |
|---|---|
| `flux-vae-bf16.safetensors` | [`Kijai/flux-fp8`](https://huggingface.co/Kijai/flux-fp8/blob/main/flux-vae-bf16.safetensors) |
| `flux2-vae.safetensors` | [`Comfy-Org/flux2-dev`](https://huggingface.co/Comfy-Org/flux2-dev/blob/main/split_files/vae/flux2-vae.safetensors) (`split_files/vae/`) |

Override with `PID_FLUX_VAE_REPO`/`PID_FLUX_VAE_PATH` (and the `FLUX2_`
equivalents) in `.env` if you host them elsewhere. The `pixel_space` **decode**
VAE is provided by the ComfyUI PiD core node — no file needed.

## 6. Scale-RAE (8×) backbone — optional / advanced

`flux` and `flux2` (the default) run fully through ComfyUI and are covered by the
steps above. The `scale_rae` 8× backbone instead runs the PiD reference CLI as a
subprocess and additionally needs:

- the gated `nvidia/PiD` checkpoint
  `checkpoints/PiD_res2k_sr8x_official_siglip_distill_4step/model_ema_bf16.pth`
  (download into `PID_MODELS_ROOT/checkpoints/...`), and
- `PID_PYTHON` pointing at a venv with PyTorch + an editable install of the
  cloned `pid-upstream` repo (`pip install -e .pidstudio/pid-upstream`).

This path is not wired into the one-click wizard yet; leave the backbone on
`flux`/`flux2` unless you specifically need single-pass 8×.

## Troubleshooting

- **`HF_TOKEN missing`** — set it in `.env` and restart the backend.
- **Download 401/403** — accept the license on the model page; check the token scope.
- **ComfyUI offline badge** — start `python main.py` inside the ComfyUI folder.
- **Output is black** — VRAM exhaustion at that resolution; lower the target
  quality (the backend guards at 10.5 MP and reports NaN/black explicitly).
- **`comfyFlux4kReady: false`** — the `pid_flux1_1024_to_4096_4step_bf16.safetensors`
  file isn't in `diffusion_models/`; re-run the wizard.

## Running tests

See [`backend/tests/README.md`](../backend/tests/README.md). The suite runs
against a live backend and skips the GPU paths automatically.
