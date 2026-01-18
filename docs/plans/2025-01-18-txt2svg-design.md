# Text-to-SVG Pipeline Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully automated CLI that converts text prompts to pen-plotter-optimized SVG files using AI image generation and centerline vectorization.

**Architecture:** 5-stage pipeline: LLM prompt enhancement → Flux.2 raster generation → skeletonization/graph extraction → vpype path optimization → SVG output. All intermediate results saved for debugging.

**Tech Stack:** Python 3.10, PyTorch (CUDA 12.x), Flux.2-dev, scikit-image, networkx, vpype, OpenRouter API

---

## Configuration

### Environment (.env)
```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
HF_TOKEN=hf_...
```

### Hardware Requirements
- NVIDIA GPU with 24GB VRAM (RTX 3090)
- CUDA 12.6

### Output Defaults
- Resolution: 1344×960 pixels (A3 proportions, divisible by 16)
- SVG dimensions: 420×297mm (A3 landscape)
- Configurable via CLI flags

---

## Project Structure

```
ml_txt2svg/
├── .env                     # API keys, model config (gitignored)
├── .env.example             # Template for .env
├── pyproject.toml           # Dependencies, Python 3.10
├── main.py                  # CLI entry point
├── modules/
│   ├── __init__.py
│   ├── prompt_engineer.py   # Stage 1: OpenRouter LLM call
│   ├── raster_generator.py  # Stage 2: Flux.2 image generation
│   ├── vectorizer.py        # Stage 3: Skeleton → Graph → Paths
│   ├── optimizer.py         # Stage 4: vpype optimization
│   └── utils.py             # Shared helpers (image I/O, debug saving)
├── tests/
│   ├── test_vectorizer.py   # Core algorithm tests
│   ├── test_prompt.py
│   └── fixtures/            # Test images (skeleton samples)
├── output/                  # Generated SVGs (gitignored)
│   └── debug/               # Intermediate files (always saved)
└── docs/
    └── plans/
```

### Debug Output Structure
```
output/debug/
├── 01_prompt_enhanced.txt   # Stage 1: The rewritten prompt
├── 02_raster_raw.png        # Stage 2: Raw Flux output
├── 02_raster_binary.png     # Stage 2: After thresholding
├── 03_skeleton.png          # Stage 3: Skeletonized image
├── 03_graph_nodes.png       # Stage 3: Overlay showing nodes (red) & edges (blue)
├── 03_graph_pruned.png      # Stage 3: After spur removal
├── 03_paths.svg             # Stage 3: Raw paths before optimization
├── 04_optimized.svg         # Stage 4: After vpype
└── stats.json               # Timing, pixel counts, path counts
```

---

## Stage 1: Prompt Engineering (OpenRouter)

### Purpose
Rewrite user's simple prompt into Flux.2-optimized prompt for line art generation.

### Flux.2 Prompting Rules
- Natural language descriptions, not keyword lists
- No negative prompts - describe what you want
- Word order matters - most important elements first
- Avoid "white background" phrase (causes blur)

### System Prompt
```
You rewrite user prompts for Flux.2 image generation,
optimized for pen plotter line art output.

Flux.2 uses natural language - write flowing descriptions, not keyword lists.
Word order matters: put the most important elements first.

Structure: Subject + Style + Details + Mood

ALWAYS frame as line art by including phrases like:
- "minimalistic line drawing" or "single continuous line art"
- "black ink on white paper" or "monochrome ink illustration"
- "clean precise lines" or "pen and ink style"
- "technical illustration" or "architectural line drawing"

DO NOT use:
- Negative phrasing ("no shading", "without color") - Flux has no negative prompts
- Keyword spam - use natural sentences instead
- "white background" phrase - causes blurry outputs

Example transformation:
Input: "a geometric skull"
Output: "Minimalistic line drawing of a geometric skull composed of
triangular facets and sharp angular planes, black ink on white paper,
technical illustration style with clean precise single-weight lines,
symmetrical front view, high contrast monochrome"

Output ONLY the rewritten prompt.
```

### Implementation
```python
from openai import OpenAI
import os

def enhance_prompt(user_prompt: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    )

    enhanced = response.choices[0].message.content.strip()
    save_debug("01_prompt_enhanced.txt",
               f"Original: {user_prompt}\n\nEnhanced: {enhanced}")

    return enhanced
```

---

## Stage 2: Raster Generation (Flux.2-dev)

### Model
- `black-forest-labs/FLUX.2-dev` from HuggingFace
- Requires `HF_TOKEN` for gated model access

### Resolution
- 1344×960 pixels (A3 proportions, both divisible by 16)
- Ratio: 1.4:1 (close to A3's 1.414:1)

### Implementation
```python
import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu

def generate_raster(prompt: str) -> tuple[Image.Image, np.ndarray]:
    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Generate
    image = pipe(
        prompt=prompt,
        width=1344,
        height=960,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]

    # Save raw output
    image.save("output/debug/02_raster_raw.png")

    # Convert to binary
    gray = np.array(image.convert("L"))
    thresh = threshold_otsu(gray)
    binary = (gray < thresh).astype(np.uint8)

    # Ensure foreground is minority (lines, not background)
    if np.mean(binary) > 0.5:
        binary = 1 - binary

    # Save binary
    Image.fromarray(binary * 255).save("output/debug/02_raster_binary.png")

    # Validate - check if image is not blank
    if np.sum(binary) < 0.01 * binary.size:
        raise ValueError("Generated image is blank or nearly blank")

    return image, binary
```

---

## Stage 3: Vectorization (The Core Algorithm)

### 3.1 Skeletonization

```python
from skimage.morphology import skeletonize
from PIL import Image

def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    # Lee's method produces smoother, better-connected skeletons
    skeleton = skeletonize(binary, method='lee')

    # Save debug
    Image.fromarray((skeleton * 255).astype(np.uint8)).save(
        "output/debug/03_skeleton.png"
    )

    return skeleton.astype(np.uint8)
```

**Why Lee's method:** Zhang-Suen can create disconnected segments at junctions. Lee (medial axis via distance transform) preserves topology better - critical for continuous pen strokes.

### 3.2 Skeleton → Graph Conversion

```python
import numpy as np
import networkx as nx
from scipy import ndimage

def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    # Step 1: Count neighbors for each pixel using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = ndimage.convolve(skeleton, kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton  # Only count skeleton pixels

    # Step 2: Identify node pixels (endpoints + junctions)
    endpoints = (neighbor_count == 1) & (skeleton == 1)    # Dead ends
    junctions = (neighbor_count >= 3) & (skeleton == 1)    # Intersections
    node_mask = endpoints | junctions

    # Step 3: Label each node with unique ID
    node_coords = np.argwhere(node_mask)  # [(y, x), ...]
    coord_to_node = {tuple(c): i for i, c in enumerate(node_coords)}

    # Step 4: Create graph, add nodes
    G = nx.Graph()
    for i, (y, x) in enumerate(node_coords):
        G.add_node(i, pos=(x, y))  # Note: (x, y) for SVG coords

    # Step 5: Trace edges between nodes
    visited_edges = set()

    for start_idx, (sy, sx) in enumerate(node_coords):
        for ny, nx_ in get_neighbors(sy, sx, skeleton):
            edge_key = frozenset([(sy, sx), (ny, nx_)])
            if edge_key in visited_edges:
                continue

            # Trace path until we hit another node
            path = [(sx, sy)]  # Store as (x, y)
            prev, curr = (sy, sx), (ny, nx_)

            while True:
                path.append((curr[1], curr[0]))  # (x, y)
                visited_edges.add(frozenset([prev, curr]))

                if node_mask[curr]:
                    # Reached another node
                    end_idx = coord_to_node[curr]
                    G.add_edge(start_idx, end_idx, pixels=path)
                    break

                # Continue tracing
                neighbors = get_neighbors(curr[0], curr[1], skeleton)
                next_pixel = [n for n in neighbors if n != prev]

                if not next_pixel:
                    break  # Dead end (shouldn't happen)

                prev, curr = curr, next_pixel[0]

    return G

def get_neighbors(y: int, x: int, skeleton: np.ndarray) -> list:
    """Return coordinates of neighboring skeleton pixels (8-connected)."""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                if skeleton[ny, nx_]:
                    neighbors.append((ny, nx_))
    return neighbors
```

### 3.3 Spur Pruning

```python
def prune_spurs(G: nx.Graph, min_length: int = 10) -> nx.Graph:
    """Remove leaf edges shorter than min_length pixels."""
    pruned = True
    while pruned:
        pruned = False
        leaves = [n for n in G.nodes() if G.degree(n) == 1]

        for leaf in leaves:
            if G.degree(leaf) == 0:
                continue

            edge = list(G.edges(leaf, data=True))[0]
            pixels = edge[2].get('pixels', [])

            if len(pixels) < min_length:
                G.remove_node(leaf)
                pruned = True  # May expose new leaves, iterate

    # Remove isolated nodes (degree 0)
    G.remove_nodes_from([n for n in G.nodes() if G.degree(n) == 0])

    return G
```

### 3.4 Path Extraction

```python
def extract_paths(G: nx.Graph) -> list[list[tuple[float, float]]]:
    """Extract all edge pixel chains as coordinate lists."""
    paths = []

    for u, v, data in G.edges(data=True):
        pixels = data.get('pixels', [])
        if len(pixels) >= 2:
            paths.append(pixels)

    return paths
```

### 3.5 Debug Visualization

```python
import cv2

def save_graph_debug(skeleton: np.ndarray, G: nx.Graph, filename: str):
    """Save visualization with nodes (red) and edges (blue)."""
    vis = cv2.cvtColor((skeleton * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw edges in blue
    for u, v, data in G.edges(data=True):
        pixels = data.get('pixels', [])
        for i in range(len(pixels) - 1):
            pt1 = (int(pixels[i][0]), int(pixels[i][1]))
            pt2 = (int(pixels[i+1][0]), int(pixels[i+1][1]))
            cv2.line(vis, pt1, pt2, (255, 0, 0), 1)

    # Draw nodes in red
    for node, data in G.nodes(data=True):
        pos = data.get('pos', (0, 0))
        cv2.circle(vis, (int(pos[0]), int(pos[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(filename, vis)
```

### 3.6 Combined Vectorization Function

```python
def raster_to_paths(binary: np.ndarray) -> list[list[tuple[float, float]]]:
    """Full vectorization pipeline: binary → skeleton → graph → paths."""

    # Skeletonize
    skeleton = skeletonize_image(binary)

    # Build graph
    G = skeleton_to_graph(skeleton)
    save_graph_debug(skeleton, G, "output/debug/03_graph_nodes.png")

    # Prune spurs
    G = prune_spurs(G, min_length=10)
    save_graph_debug(skeleton, G, "output/debug/03_graph_pruned.png")

    # Extract paths
    paths = extract_paths(G)

    return paths
```

---

## Stage 4: vpype Optimization

### Operations
| Operation | Purpose | Tolerance |
|-----------|---------|-----------|
| `linemerge` | Connect nearby endpoints into continuous strokes | 0.1mm |
| `linesimplify` | Reduce vertices, smooth pixel jitter | 0.05mm |
| `linesort` | TSP solver minimizes pen-up travel time | - |
| `reloop` | Align loop start/end for clean closure | 0.1mm |

### Implementation
```python
import vpype as vp
from pathlib import Path

def optimize_paths(paths: list[list[tuple[float, float]]],
                   width_mm: float,
                   height_mm: float,
                   source_width_px: int,
                   source_height_px: int) -> vp.Document:
    # Create vpype document
    doc = vp.Document()
    lc = vp.LineCollection()

    # Scale factor: pixels → mm
    scale_x = width_mm / source_width_px
    scale_y = height_mm / source_height_px

    # Convert paths to vpype lines (complex numbers: x + yj)
    for path in paths:
        if len(path) < 2:
            continue
        line = [complex(x * scale_x, y * scale_y) for x, y in path]
        lc.append(line)

    doc.add(lc, layer_id=1)

    # Save pre-optimization debug
    vp.write_svg(Path("output/debug/03_paths.svg"), doc)

    # Optimization pipeline
    doc = vp.linemerge(doc, tolerance="0.1mm")
    doc = vp.linesimplify(doc, tolerance="0.05mm")
    doc = vp.linesort(doc)
    doc = vp.reloop(doc, tolerance="0.1mm")

    # Save post-optimization debug
    vp.write_svg(Path("output/debug/04_optimized.svg"), doc)

    return doc

def save_final_svg(doc: vp.Document, output_path: Path,
                   width_mm: float, height_mm: float):
    vp.write_svg(
        output_path,
        doc,
        page_size=(f"{width_mm}mm", f"{height_mm}mm"),
        center=True,
    )
```

---

## Stage 5: CLI Entry Point (main.py)

```python
#!/usr/bin/env python3
"""Text-to-SVG pipeline for AxiDraw pen plotters."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from modules.prompt_engineer import enhance_prompt
from modules.raster_generator import generate_raster
from modules.vectorizer import raster_to_paths
from modules.optimizer import optimize_paths, save_final_svg
from modules.utils import setup_output_dirs, save_debug

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate plotter-ready SVG from text prompt"
    )
    parser.add_argument("prompt", help="Text description of desired image")
    parser.add_argument("--width", type=float, default=420,
                        help="Output width in mm (default: 420 for A3)")
    parser.add_argument("--height", type=float, default=297,
                        help="Output height in mm (default: 297 for A3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: auto-timestamped)")
    parser.add_argument("--skip-enhance", action="store_true",
                        help="Skip LLM prompt enhancement, use raw prompt")

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = setup_output_dirs()
    stats = {"timestamp": timestamp, "stages": {}}

    # Stage 1: Prompt Enhancement
    t0 = time.time()
    if args.skip_enhance:
        enhanced_prompt = args.prompt
        save_debug("01_prompt_enhanced.txt", f"Original (no enhancement): {args.prompt}")
    else:
        enhanced_prompt = enhance_prompt(args.prompt)
    stats["stages"]["prompt"] = {"time": time.time() - t0}
    print(f"[1/5] Prompt: {enhanced_prompt[:80]}...")

    # Stage 2: Raster Generation
    t0 = time.time()
    raster, binary = generate_raster(enhanced_prompt)
    stats["stages"]["raster"] = {"time": time.time() - t0}
    print(f"[2/5] Raster generated: {binary.shape}")

    # Stage 3: Vectorization
    t0 = time.time()
    paths = raster_to_paths(binary)
    stats["stages"]["vectorize"] = {
        "time": time.time() - t0,
        "path_count": len(paths),
        "total_points": sum(len(p) for p in paths),
    }
    print(f"[3/5] Vectorized: {len(paths)} paths")

    # Stage 4: Optimization
    t0 = time.time()
    doc = optimize_paths(
        paths,
        args.width, args.height,
        binary.shape[1], binary.shape[0]
    )
    stats["stages"]["optimize"] = {"time": time.time() - t0}
    print(f"[4/5] Optimized paths")

    # Stage 5: Output
    output_name = args.output or f"output_{timestamp}.svg"
    output_path = output_dir / output_name
    save_final_svg(doc, output_path, args.width, args.height)

    # Save stats
    stats["total_time"] = sum(s["time"] for s in stats["stages"].values())
    save_debug("stats.json", json.dumps(stats, indent=2))

    print(f"[5/5] Saved: {output_path}")
    print(f"      Debug files: output/debug/")
    print(f"      Total time: {stats['total_time']:.1f}s")

if __name__ == "__main__":
    main()
```

### CLI Usage

```bash
# Basic usage (A3 default)
python main.py "a geometric skull"

# Custom dimensions
python main.py "circuit board pattern" --width 297 --height 210

# Skip LLM enhancement (use exact prompt)
python main.py "minimalistic line drawing of a cat" --skip-enhance

# Custom output name
python main.py "mountain landscape" --output mountains.svg
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "txt2svg"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "scikit-image",
    "opencv-python",
    "numpy",
    "Pillow",
    "networkx",
    "scipy",
    "vpype",
    "openai",
    "python-dotenv",
]

[project.optional-dependencies]
dev = [
    "pytest",
]
```

### Installation Notes
- PyTorch with CUDA 12.x: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Flux.2-dev requires HuggingFace token with model access approval

---

## Error Handling

| Condition | Response |
|-----------|----------|
| Blank/low-contrast image | Raise `ValueError` with message, save debug files |
| OpenRouter API failure | Raise with error details, suggest checking API key |
| CUDA out of memory | Log error, suggest reducing resolution |
| Empty graph (no paths) | Raise `ValueError`, check binary threshold |

All errors save intermediate debug files before failing.
