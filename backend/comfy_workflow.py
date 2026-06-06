"""
PiD ComfyUI workflow template.

Instead of the NVIDIA reference CLI (`from_clean_flux`), this uses ComfyUI's
native PiD nodes (`PiDConditioning`, `pixel_space` VAE, `pixeldit` CLIP). Why:
  - The model stays resident in ComfyUI's RAM → the second upscale is instant.
  - Tiled pixel-space VAE decode → high resolution stays VRAM-safe.
  - Local text encoder (gemma_2_2b_it_elm) → no online HF download per run.

Based on the "02" workflow (pure image→upscale, no text-to-image). Emits the
ComfyUI `/prompt` (execution) format — NOT the UI graph format. Node schemas
are verifiable with `/object_info` (PiDConditioning input names can vary across
ComfyUI versions).

Flow:
  LoadImage → VAEEncode(flux-vae) → PiDConditioning(backbone)
    → SamplerCustom(sampler=lcm, sigmas=PiD distill t-list, cfg=1, add_noise)
    → VAEDecode(pixel_space) → SaveImage
  target latent: EmptyChromaRadianceLatentImage(W×H)
"""

from __future__ import annotations

from typing import Any, Optional

# PiD distill sampler sigma schedules (ManualSigmas string). 4-step is full
# quality (the NVIDIA distill reference); 1/2-step for speed are subsets of the
# 4-step points — fast preview (quality drops a little, calibrated to 4-step).
PID_SIGMA_SCHEDULES = {
    4: "0.999,0.866,0.634,0.342,0",
    2: "0.999,0.634,0",
    1: "0.999,0",
}
PID_SIGMAS = PID_SIGMA_SCHEDULES[4]


def sigmas_for_steps(inference_steps: int) -> str:
    """1/2/4 steps → ManualSigmas string. Unknown values fall back to 4-step."""
    return PID_SIGMA_SCHEDULES.get(int(inference_steps), PID_SIGMA_SCHEDULES[4])


# Backbone + target resolution → ComfyUI diffusion_models filename.
# The 512→2048 model expects 512 input (2K out); 1024→4096 expects 1024 (4K).
PID_MODEL_FILES = {
    ("flux", "2k"): "pid_flux1_512_to_2048_4step_bf16.safetensors",
    ("flux", "4k"): "pid_flux1_1024_to_4096_4step_bf16.safetensors",
    ("flux2", "2k"): "pid_flux2_512_to_2048_4step_bf16.safetensors",
    ("flux2", "4k"): "pid_flux2_1024_to_4096_4step_bf16.safetensors",
}

# PiDConditioning.latent_format combo value. Flux1 (16-ch) AND Flux2 (128-ch)
# both use "flux" → the node auto-detects from the channel dim. So the flux vs
# flux2 difference is NOT in latent_format, it's in the VAEEncode VAE.
PID_LATENT_FORMAT = {
    "flux": "flux",
    "flux2": "flux",
}

# VAE file each backbone uses for VAEEncode (in ComfyUI models/vae/).
PID_ENCODE_VAE = {
    "flux": "flux-vae-bf16.safetensors",
    "flux2": "flux2-vae.safetensors",
}

# Text encoder (local, no HF download at run time).
PID_CLIP_NAME = "gemma_2_2b_it_elm_bf16.safetensors"
PID_NEGATIVE = "low quality, worst quality, over-saturated, blurry, deformed, watermark"
PID_DEFAULT_PROMPT = "a high quality photo, sharp focus, detailed"


def model_file(backbone: str, target_quality: str) -> str:
    """backbone + 2k/4k/8k → model file. 8k → the 4k model (the dual pass is
    handled by the caller; a single pass is 1024→4096)."""
    q = "2k" if target_quality == "2k" else "4k"
    key = (backbone if backbone in ("flux", "flux2") else "flux", q)
    return PID_MODEL_FILES[key]


def target_dims(target_quality: str, aspect: float) -> tuple[int, int]:
    """Target output size (W, H), aspect-preserving, rounded to a multiple of 16.
    2k → long edge 2048, 4k → 4096. aspect = w/h."""
    long_edge = 2048 if target_quality == "2k" else 4096
    if aspect >= 1.0:
        w, h = long_edge, round(long_edge / aspect)
    else:
        w, h = round(long_edge * aspect), long_edge
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    return w, h


def build_workflow(
    *,
    image_filename: str,
    backbone: str,
    target_quality: str,
    out_w: int,
    out_h: int,
    seed: int,
    prompt: Optional[str],
    degrade_sigma: float,
    filename_prefix: str,
    inference_steps: int = 4,
) -> dict[str, Any]:
    """Builds a workflow dict in ComfyUI /prompt 'prompt' (execution) format.

    PiDConditioning schema is LOCKED from nodes_pid.py:
      positive(CONDITIONING), latent(LATENT),
      latent_format(Combo: flux|sd3|sdxl|qwenimage), degrade_sigma(Float 0-1).
    """
    pos_text = (prompt or "").strip() or PID_DEFAULT_PROMPT
    model_name = model_file(backbone, target_quality)
    encode_vae = PID_ENCODE_VAE.get(backbone, "flux-vae-bf16.safetensors")
    latent_format = PID_LATENT_FORMAT.get(backbone, "flux")

    pid_cond_inputs: dict[str, Any] = {
        "positive": ["pos", 0],
        "latent": ["vae_encode", 0],
        "latent_format": latent_format,
        "degrade_sigma": float(degrade_sigma),
    }

    return {
        "vae_load": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": encode_vae},
        },
        "load_img": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "vae_encode": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["load_img", 0], "vae": ["vae_load", 0]},
        },
        "clip_load": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": PID_CLIP_NAME,
                "type": "pixeldit",
                "device": "default",
            },
        },
        "pos": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_load", 0], "text": pos_text},
        },
        "neg": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_load", 0], "text": PID_NEGATIVE},
        },
        "neg_zero": {
            # The 02 workflow doesn't ZeroOut the negative, but at distill cfg=1
            # the negative is unused; a plain CLIPTextEncode is enough. (ZeroOut
            # is optional belt-and-braces.)
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["neg", 0]},
        },
        "pid_cond": {
            "class_type": "PiDConditioning",
            "inputs": pid_cond_inputs,
        },
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
        },
        "empty": {
            "class_type": "EmptyChromaRadianceLatentImage",
            "inputs": {"width": out_w, "height": out_h, "batch_size": 1},
        },
        "sampler_sel": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "lcm"},
        },
        "sigmas": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": sigmas_for_steps(inference_steps)},
        },
        "sampler": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": int(seed),
                "cfg": 1.0,
                "model": ["unet", 0],
                "positive": ["pid_cond", 0],
                "negative": ["neg_zero", 0],
                "sampler": ["sampler_sel", 0],
                "sigmas": ["sigmas", 0],
                "latent_image": ["empty", 0],
            },
        },
        "pixel_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "pixel_space"},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["pixel_vae", 0]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
        },
    }
