"""Stage 2: Raster image generation using Flux.2-dev (4-bit quantized)."""

from pathlib import Path

import numpy as np
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from PIL import Image
from skimage.filters import threshold_otsu
from transformers import Mistral3ForConditionalGeneration

# Default resolution (A3 proportions, divisible by 16)
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 960

# Quantized model repo for 24GB VRAM GPUs
QUANTIZED_REPO = "diffusers/FLUX.2-dev-bnb-4bit"

# Cached pipeline (loaded once, reused across calls)
_cached_pipe = None


def _get_pipeline() -> Flux2Pipeline:
    """Get or create the cached Flux pipeline."""
    global _cached_pipe

    if _cached_pipe is not None:
        return _cached_pipe

    torch_dtype = torch.bfloat16

    # Load 4-bit quantized transformer
    transformer = Flux2Transformer2DModel.from_pretrained(
        QUANTIZED_REPO,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    # Load 4-bit quantized text encoder (Mistral-3)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        QUANTIZED_REPO,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    # Load pipeline with quantized components
    pipe = Flux2Pipeline.from_pretrained(
        QUANTIZED_REPO,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.enable_model_cpu_offload()

    _cached_pipe = pipe
    return _cached_pipe


def generate_raster(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    seed: int | None = None,
    debug_dir: Path | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a raster image from a text prompt using Flux.2-dev (4-bit).

    Uses 4-bit quantized transformer and text encoder to fit in 24GB VRAM.

    Args:
        prompt: The text prompt for image generation.
        width: Output width in pixels (must be divisible by 16).
        height: Output height in pixels (must be divisible by 16).
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for generation.
        seed: Random seed for reproducible generation (None for random).
        debug_dir: Directory to save debug files (None to skip debug output).

    Returns:
        Tuple of (PIL Image, binary numpy array).

    Raises:
        ValueError: If generated image is blank or nearly blank.
    """
    # Get cached pipeline (only loads once)
    pipe = _get_pipeline()

    # Create generator for reproducible results
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # Generate
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Save raw output
    if debug_dir:
        image.save(debug_dir / "02_raster_raw.png")

    # Convert to binary
    gray = np.array(image.convert("L"))
    thresh = threshold_otsu(gray)
    binary = (gray < thresh).astype(np.uint8)

    # Ensure foreground is minority (lines, not background)
    if np.mean(binary) > 0.5:
        binary = 1 - binary

    # Save binary
    if debug_dir:
        Image.fromarray(binary * 255).save(debug_dir / "02_raster_binary.png")

    # Validate - check if image is not blank
    if np.sum(binary) < 0.01 * binary.size:
        raise ValueError("Generated image is blank or nearly blank")

    return image, binary
