# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

txt2plotter converts text prompts to pen-plotter-ready SVG files through a 5-stage pipeline:
1. **Prompt Enhancement** (`modules/prompt_engineer.py`) - LLM via OpenRouter rewrites prompts for optimal Flux.2 line art
2. **Raster Generation** (`modules/raster_generator.py`) - FLUX.2 [dev] generates high-contrast line art. Defaults to remote inference via fal.ai; a local 4-bit quantized backend is available for 24GB-VRAM GPUs.
3. **Vectorization** (`modules/vectorizer.py`) - Skeletonization + graph extraction produces clean paths
4. **Optimization** (`modules/optimizer.py`) - vpype merges, simplifies, and sorts paths for efficient plotting
5. **Output** - Final SVG with configurable dimensions

## Commands

```bash
# Install dependencies (remote fal.ai backend, no GPU needed)
pip install -e .

# Install with local GPU backend (24GB VRAM required)
pip install -e ".[local-gpu]"

# Basic run (A3 size, uses fal.ai by default)
python main.py "a geometric skull"

# Use the local GPU backend instead of fal.ai
python main.py "a geometric skull" --local-flux

# Custom dimensions
python main.py "circuit board pattern" --width 297 --height 210

# Multiple variations
python main.py "mountain landscape" -n 5

# Reproducible with seed
python main.py "geometric pattern" --seed 42

# Skip LLM prompt enhancement
python main.py "minimalistic line drawing of a cat" --skip-enhance

# Batch mode from file
python main.py --batch prompts.txt -n 10
```

## Architecture Notes

**Pipeline flow in `main.py`**: Each stage passes data to the next. Raster generator returns both raw PIL Image and binary numpy array. Vectorizer takes binary, returns list of paths. Optimizer takes paths and dimensions, returns vpype Document.

**Raster backends**: `raster_generator.py` dispatches between a fal.ai remote backend (default, `fal-ai/flux-2`) and a local 4-bit quantized Flux backend. Select via `--local-flux` CLI flag or `TXT2PLOTTER_BACKEND` env var. The local backend uses a module-level `_cached_pipe` singleton so the heavy pipeline loads once per process. GPU-only imports (`torch`, `diffusers`, `transformers`) are lazy so the fal backend works without them installed.

**Graph-based vectorization**: The vectorizer builds a NetworkX graph where nodes are endpoints/junctions and edges store pixel paths. This enables spur pruning and clean path extraction.

**Coordinate systems**: Pixel coordinates (y, x) are converted to SVG coordinates (x, y) during graph building. The optimizer scales from pixels to millimeters.

## Environment Variables

- `OPENROUTER_API_KEY` - Required for prompt enhancement
- `OPENROUTER_MODEL` - LLM model (default: `openai/gpt-4o-mini`)
- `FAL_KEY` - Required for the default fal.ai raster backend
- `TXT2PLOTTER_BACKEND` - Optional: `fal` (default) or `local`. CLI `--local-flux` overrides.
- `HF_TOKEN` - Only required when using `--local-flux` (gated Flux.2-dev model access)

## Output Structure

- `output/*.svg` - Final plotter-ready SVGs
- `output/debug/` - Intermediate files (enhanced prompt, raw/binary rasters, skeleton, graph visualizations, path SVGs)
- Batch mode organizes by prompt: `output/<prompt_slug>/`
