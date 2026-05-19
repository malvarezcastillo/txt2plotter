"""Stage 2: Raster image generation via fal.ai (default) or local Flux.2-dev."""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

# Default resolution (A3 proportions, divisible by 16)
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 960

FAL_MODEL = "fal-ai/flux-2"

# Cached local pipeline (loaded once, reused across calls). Only used by local backend.
_cached_pipe = None


def _resolve_backend(backend: str | None) -> str:
    if backend is not None:
        return backend
    return os.getenv("TXT2PLOTTER_BACKEND", "fal").lower()


def _generate_via_fal(
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
) -> Image.Image:
    """Generate an image using the fal.ai FLUX.2 [dev] endpoint."""
    if not os.getenv("FAL_KEY"):
        raise RuntimeError(
            "FAL_KEY is not set. Add it to your .env file (see .env.example) "
            "or pass --local-flux to use the local GPU backend."
        )

    import fal_client
    import requests

    arguments: dict = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "enable_safety_checker": False,
        "output_format": "png",
    }
    if seed is not None:
        arguments["seed"] = seed

    result = fal_client.subscribe(FAL_MODEL, arguments=arguments, with_logs=False)

    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"fal.ai returned no images. Response: {result}")
    image_url = images[0]["url"]

    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _get_local_pipeline():
    """Get or create the cached local Flux pipeline (requires GPU deps)."""
    global _cached_pipe

    if _cached_pipe is not None:
        return _cached_pipe

    try:
        import torch
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            "Local backend requires GPU dependencies. Install with: "
            "pip install -e '.[local-gpu]'"
        ) from e

    quantized_repo = "diffusers/FLUX.2-dev-bnb-4bit"
    torch_dtype = torch.bfloat16

    transformer = Flux2Transformer2DModel.from_pretrained(
        quantized_repo,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        quantized_repo,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    pipe = Flux2Pipeline.from_pretrained(
        quantized_repo,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.enable_model_cpu_offload()

    _cached_pipe = pipe
    return _cached_pipe


def _generate_via_local(
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int | None,
) -> Image.Image:
    """Generate an image using local 4-bit quantized Flux.2-dev."""
    import torch

    pipe = _get_local_pipeline()

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    return pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]


def generate_raster(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    seed: int | None = None,
    debug_dir: Path | None = None,
    backend: str | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a raster image from a text prompt.

    Args:
        prompt: The text prompt for image generation.
        width: Output width in pixels (must be divisible by 16).
        height: Output height in pixels (must be divisible by 16).
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for generation.
        seed: Random seed for reproducible generation (None for random).
        debug_dir: Directory to save debug files (None to skip debug output).
        backend: "fal" (default) or "local". If None, reads TXT2PLOTTER_BACKEND env var.

    Returns:
        Tuple of (PIL Image, binary numpy array).

    Raises:
        ValueError: If generated image is blank or nearly blank.
        RuntimeError: If the selected backend is misconfigured.
    """
    resolved = _resolve_backend(backend)
    if resolved == "fal":
        image = _generate_via_fal(
            prompt, width, height, num_inference_steps, guidance_scale, seed
        )
    elif resolved == "local":
        image = _generate_via_local(
            prompt, width, height, num_inference_steps, guidance_scale, seed
        )
    else:
        raise ValueError(f"Unknown backend: {resolved!r} (expected 'fal' or 'local')")

    if debug_dir:
        image.save(debug_dir / "02_raster_raw.png")

    gray = np.array(image.convert("L"))
    thresh = threshold_otsu(gray)
    binary = (gray < thresh).astype(np.uint8)

    if np.mean(binary) > 0.5:
        binary = 1 - binary

    if debug_dir:
        Image.fromarray(binary * 255).save(debug_dir / "02_raster_binary.png")

    if np.sum(binary) < 0.01 * binary.size:
        raise ValueError("Generated image is blank or nearly blank")

    return image, binary
