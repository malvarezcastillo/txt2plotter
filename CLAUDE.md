# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

txt2plotter converts text prompts to pen-plotter-ready SVG files through a 5-stage pipeline:
1. **Prompt Enhancement** (`modules/prompt_engineer.py`) - LLM via OpenRouter rewrites prompts for optimal Flux.2 line art
2. **Raster Generation** (`modules/raster_generator.py`) - Flux.2-dev (4-bit quantized) generates high-contrast line art
3. **Vectorization** (`modules/vectorizer.py`) - Skeletonization + graph extraction produces clean paths
4. **Optimization** (`modules/optimizer.py`) - vpype merges, simplifies, and sorts paths for efficient plotting
5. **Output** - Final SVG with configurable dimensions

## Commands

```bash
# Install dependencies
pip install -e .

# Basic run (A3 size)
python main.py "a geometric skull"

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

**Flux pipeline caching**: `raster_generator.py` uses a module-level `_cached_pipe` singleton. Pipeline loads once on first call, reuses for subsequent generations.

**Graph-based vectorization**: The vectorizer builds a NetworkX graph where nodes are endpoints/junctions and edges store pixel paths. This enables spur pruning and clean path extraction.

**Coordinate systems**: Pixel coordinates (y, x) are converted to SVG coordinates (x, y) during graph building. The optimizer scales from pixels to millimeters.

## Environment Variables

- `OPENROUTER_API_KEY` - Required for prompt enhancement
- `OPENROUTER_MODEL` - LLM model (default: `openai/gpt-4o-mini`)
- `HF_TOKEN` - Required for Flux.2-dev gated model access

## Output Structure

- `output/*.svg` - Final plotter-ready SVGs
- `output/debug/` - Intermediate files (enhanced prompt, raw/binary rasters, skeleton, graph visualizations, path SVGs)
- Batch mode organizes by prompt: `output/<prompt_slug>/`
