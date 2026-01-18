# Text-to-SVG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI that converts text prompts to pen-plotter-optimized SVG files using Flux.2 and centerline vectorization.

**Architecture:** 5-stage pipeline (prompt enhancement → raster generation → skeletonization/graph → vpype optimization → SVG output) with debug output at every stage.

**Tech Stack:** Python 3.10, PyTorch (CUDA 12.x), Flux.2-dev, scikit-image, networkx, vpype, OpenRouter API

---

## Task 0: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `modules/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

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

**Step 2: Create .env.example**

```
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
HF_TOKEN=hf_your-token-here
```

**Step 3: Create .gitignore**

```
.env
__pycache__/
*.pyc
.venv/
venv/
output/
*.egg-info/
.pytest_cache/
```

**Step 4: Create empty module files**

```bash
mkdir -p modules tests tests/fixtures output/debug
touch modules/__init__.py tests/__init__.py
```

**Step 5: Create and activate virtual environment**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**Step 6: Install dependencies**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

**Step 7: Verify installation**

Run: `python -c "import torch; print(torch.cuda.is_available())"`
Expected: `True`

**Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore modules/ tests/
git commit -m "chore: project setup with dependencies"
```

---

## Task 1: Utils Module (Debug Helpers)

**Files:**
- Create: `modules/utils.py`
- Create: `tests/test_utils.py`

**Step 1: Write failing test for setup_output_dirs**

```python
# tests/test_utils.py
import os
import shutil
from pathlib import Path

def test_setup_output_dirs_creates_directories():
    # Clean up if exists
    if Path("output").exists():
        shutil.rmtree("output")

    from modules.utils import setup_output_dirs

    result = setup_output_dirs()

    assert result == Path("output")
    assert Path("output").exists()
    assert Path("output/debug").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py::test_setup_output_dirs_creates_directories -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# modules/utils.py
from pathlib import Path


def setup_output_dirs() -> Path:
    """Create output directories, return output path."""
    output_dir = Path("output")
    debug_dir = output_dir / "debug"

    output_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    return output_dir
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py::test_setup_output_dirs_creates_directories -v`
Expected: PASS

**Step 5: Write failing test for save_debug**

```python
# tests/test_utils.py (append)

def test_save_debug_writes_text_file():
    from modules.utils import setup_output_dirs, save_debug

    setup_output_dirs()
    save_debug("test_output.txt", "Hello, world!")

    path = Path("output/debug/test_output.txt")
    assert path.exists()
    assert path.read_text() == "Hello, world!"
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_utils.py::test_save_debug_writes_text_file -v`
Expected: FAIL with "ImportError" (save_debug not found)

**Step 7: Implement save_debug**

```python
# modules/utils.py (append)


def save_debug(filename: str, content: str) -> Path:
    """Save debug content to output/debug/filename."""
    path = Path("output/debug") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_utils.py::test_save_debug_writes_text_file -v`
Expected: PASS

**Step 9: Commit**

```bash
git add modules/utils.py tests/test_utils.py
git commit -m "feat: add utils module with debug helpers"
```

---

## Task 2: Vectorizer - Skeletonization

**Files:**
- Create: `modules/vectorizer.py`
- Create: `tests/test_vectorizer.py`
- Create: `tests/fixtures/simple_cross.png` (test image)

**Step 1: Create test fixture (simple cross image)**

```python
# Run this once to create fixture
import numpy as np
from PIL import Image

# Create 100x100 image with a cross shape
img = np.zeros((100, 100), dtype=np.uint8)
img[45:55, 20:80] = 1  # Horizontal bar
img[20:80, 45:55] = 1  # Vertical bar
Image.fromarray(img * 255).save("tests/fixtures/simple_cross.png")
```

**Step 2: Write failing test for skeletonize_image**

```python
# tests/test_vectorizer.py
import numpy as np
from pathlib import Path


def test_skeletonize_image_produces_thin_lines():
    from modules.vectorizer import skeletonize_image
    from modules.utils import setup_output_dirs
    from PIL import Image

    setup_output_dirs()

    # Load test fixture
    img = np.array(Image.open("tests/fixtures/simple_cross.png").convert("L"))
    binary = (img > 127).astype(np.uint8)

    skeleton = skeletonize_image(binary)

    # Skeleton should be thinner than original
    assert skeleton.sum() < binary.sum()
    # Skeleton should still have content
    assert skeleton.sum() > 0
    # Skeleton should be binary
    assert set(np.unique(skeleton)).issubset({0, 1})
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_skeletonize_image_produces_thin_lines -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# modules/vectorizer.py
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def skeletonize_image(binary: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary image using Lee's method.

    Args:
        binary: Binary image (0 and 1 values, foreground=1)

    Returns:
        Skeletonized image (0 and 1 values)
    """
    # Lee's method produces smoother, better-connected skeletons
    skeleton = skeletonize(binary.astype(bool), method='lee')

    # Save debug output
    Image.fromarray((skeleton * 255).astype(np.uint8)).save(
        "output/debug/03_skeleton.png"
    )

    return skeleton.astype(np.uint8)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_vectorizer.py::test_skeletonize_image_produces_thin_lines -v`
Expected: PASS

**Step 6: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py tests/fixtures/
git commit -m "feat: add skeletonization using Lee's method"
```

---

## Task 3: Vectorizer - Neighbor Counting

**Files:**
- Modify: `modules/vectorizer.py`
- Modify: `tests/test_vectorizer.py`

**Step 1: Write failing test for get_neighbors**

```python
# tests/test_vectorizer.py (append)

def test_get_neighbors_returns_8_connected():
    from modules.vectorizer import get_neighbors

    # 3x3 grid, all ones
    skeleton = np.ones((3, 3), dtype=np.uint8)

    # Center pixel should have 8 neighbors
    neighbors = get_neighbors(1, 1, skeleton)
    assert len(neighbors) == 8

    # Corner pixel should have 3 neighbors
    neighbors = get_neighbors(0, 0, skeleton)
    assert len(neighbors) == 3


def test_get_neighbors_respects_skeleton():
    from modules.vectorizer import get_neighbors

    # Only center and right neighbor are skeleton pixels
    skeleton = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ], dtype=np.uint8)

    # Center has only 1 neighbor (to the right)
    neighbors = get_neighbors(1, 1, skeleton)
    assert len(neighbors) == 1
    assert neighbors[0] == (1, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_get_neighbors_returns_8_connected -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
# modules/vectorizer.py (append)


def get_neighbors(y: int, x: int, skeleton: np.ndarray) -> list[tuple[int, int]]:
    """
    Return coordinates of neighboring skeleton pixels (8-connected).

    Args:
        y: Row index
        x: Column index
        skeleton: Binary skeleton image

    Returns:
        List of (y, x) tuples for neighboring skeleton pixels
    """
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx]:
                    neighbors.append((ny, nx))
    return neighbors
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectorizer.py -v -k "neighbors"`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py
git commit -m "feat: add get_neighbors for 8-connected pixel lookup"
```

---

## Task 4: Vectorizer - Skeleton to Graph

**Files:**
- Modify: `modules/vectorizer.py`
- Modify: `tests/test_vectorizer.py`

**Step 1: Write failing test for skeleton_to_graph**

```python
# tests/test_vectorizer.py (append)

def test_skeleton_to_graph_simple_line():
    from modules.vectorizer import skeleton_to_graph

    # Horizontal line: 5 pixels
    skeleton = np.zeros((5, 10), dtype=np.uint8)
    skeleton[2, 2:7] = 1  # 5 pixels in a row

    G = skeleton_to_graph(skeleton)

    # Should have 2 nodes (endpoints) and 1 edge
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1

    # Edge should have pixel data
    edge_data = list(G.edges(data=True))[0]
    pixels = edge_data[2]['pixels']
    assert len(pixels) == 5


def test_skeleton_to_graph_cross():
    from modules.vectorizer import skeleton_to_graph

    # Cross shape: 1 junction, 4 endpoints
    skeleton = np.zeros((7, 7), dtype=np.uint8)
    skeleton[3, 1:6] = 1  # Horizontal
    skeleton[1:6, 3] = 1  # Vertical

    G = skeleton_to_graph(skeleton)

    # Should have 5 nodes: 4 endpoints + 1 junction
    assert G.number_of_nodes() == 5
    # Should have 4 edges (one per arm)
    assert G.number_of_edges() == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_skeleton_to_graph_simple_line -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
# modules/vectorizer.py (add imports at top)
import networkx as nx
from scipy import ndimage


# modules/vectorizer.py (append function)

def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Convert a skeleton image to a NetworkX graph.

    Nodes are endpoints (1 neighbor) or junctions (3+ neighbors).
    Edges are pixel chains connecting nodes.

    Args:
        skeleton: Binary skeleton image (0 and 1 values)

    Returns:
        NetworkX graph with 'pos' attribute on nodes (x, y)
        and 'pixels' attribute on edges [(x, y), ...]
    """
    # Count neighbors for each pixel using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    neighbor_count = neighbor_count * skeleton  # Only count skeleton pixels

    # Identify node pixels (endpoints + junctions)
    endpoints = (neighbor_count == 1) & (skeleton == 1)
    junctions = (neighbor_count >= 3) & (skeleton == 1)
    node_mask = endpoints | junctions

    # Label each node with unique ID
    node_coords = np.argwhere(node_mask)  # [(y, x), ...]
    coord_to_node = {tuple(c): i for i, c in enumerate(node_coords)}

    # Create graph, add nodes
    G = nx.Graph()
    for i, (y, x) in enumerate(node_coords):
        G.add_node(i, pos=(x, y))  # (x, y) for SVG coords

    # Trace edges between nodes
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
                next_pixels = [n for n in neighbors if n != prev]

                if not next_pixels:
                    break  # Dead end (shouldn't happen in clean skeleton)

                prev, curr = curr, next_pixels[0]

    return G
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectorizer.py -v -k "graph"`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py
git commit -m "feat: add skeleton_to_graph conversion"
```

---

## Task 5: Vectorizer - Spur Pruning

**Files:**
- Modify: `modules/vectorizer.py`
- Modify: `tests/test_vectorizer.py`

**Step 1: Write failing test for prune_spurs**

```python
# tests/test_vectorizer.py (append)

def test_prune_spurs_removes_short_branches():
    from modules.vectorizer import skeleton_to_graph, prune_spurs

    # Line with a short spur
    skeleton = np.zeros((10, 20), dtype=np.uint8)
    skeleton[5, 2:18] = 1  # Main horizontal line (16 px)
    skeleton[3:5, 10] = 1  # Short spur up (2 px)

    G = skeleton_to_graph(skeleton)
    original_edges = G.number_of_edges()

    G = prune_spurs(G, min_length=5)

    # Spur should be removed
    assert G.number_of_edges() < original_edges


def test_prune_spurs_keeps_long_branches():
    from modules.vectorizer import skeleton_to_graph, prune_spurs

    # Cross with long arms
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 2:18] = 1  # Horizontal (16 px)
    skeleton[2:18, 10] = 1  # Vertical (16 px)

    G = skeleton_to_graph(skeleton)
    original_edges = G.number_of_edges()

    G = prune_spurs(G, min_length=5)

    # All edges should remain
    assert G.number_of_edges() == original_edges
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_prune_spurs_removes_short_branches -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
# modules/vectorizer.py (append)


def prune_spurs(G: nx.Graph, min_length: int = 10) -> nx.Graph:
    """
    Remove leaf edges shorter than min_length pixels.

    Args:
        G: NetworkX graph from skeleton_to_graph
        min_length: Minimum pixel count to keep an edge

    Returns:
        Pruned graph (modified in place)
    """
    pruned = True
    while pruned:
        pruned = False
        leaves = [n for n in G.nodes() if G.degree(n) == 1]

        for leaf in leaves:
            if G.degree(leaf) == 0:
                continue

            edges = list(G.edges(leaf, data=True))
            if not edges:
                continue

            edge = edges[0]
            pixels = edge[2].get('pixels', [])

            if len(pixels) < min_length:
                G.remove_node(leaf)
                pruned = True  # May expose new leaves, iterate

    # Remove isolated nodes (degree 0)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    return G
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectorizer.py -v -k "prune"`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py
git commit -m "feat: add spur pruning for noise removal"
```

---

## Task 6: Vectorizer - Path Extraction & Debug Visualization

**Files:**
- Modify: `modules/vectorizer.py`
- Modify: `tests/test_vectorizer.py`

**Step 1: Write failing test for extract_paths**

```python
# tests/test_vectorizer.py (append)

def test_extract_paths_returns_coordinate_lists():
    from modules.vectorizer import skeleton_to_graph, extract_paths

    # Simple horizontal line
    skeleton = np.zeros((5, 10), dtype=np.uint8)
    skeleton[2, 2:8] = 1  # 6 pixels

    G = skeleton_to_graph(skeleton)
    paths = extract_paths(G)

    assert len(paths) == 1
    assert len(paths[0]) == 6
    # Each point should be (x, y) tuple
    assert all(len(p) == 2 for p in paths[0])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_extract_paths_returns_coordinate_lists -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation for extract_paths**

```python
# modules/vectorizer.py (append)


def extract_paths(G: nx.Graph) -> list[list[tuple[float, float]]]:
    """
    Extract all edge pixel chains as coordinate lists.

    Args:
        G: NetworkX graph with 'pixels' edge attribute

    Returns:
        List of paths, where each path is [(x, y), ...]
    """
    paths = []

    for u, v, data in G.edges(data=True):
        pixels = data.get('pixels', [])
        if len(pixels) >= 2:
            paths.append(pixels)

    return paths
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vectorizer.py::test_extract_paths_returns_coordinate_lists -v`
Expected: PASS

**Step 5: Add debug visualization function**

```python
# modules/vectorizer.py (add import at top)
import cv2


# modules/vectorizer.py (append)


def save_graph_debug(skeleton: np.ndarray, G: nx.Graph, filename: str) -> None:
    """
    Save visualization with nodes (red) and edges (blue).

    Args:
        skeleton: Original skeleton image for background
        G: NetworkX graph to visualize
        filename: Output path (e.g., "output/debug/03_graph_nodes.png")
    """
    vis = cv2.cvtColor((skeleton * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw edges in blue
    for u, v, data in G.edges(data=True):
        pixels = data.get('pixels', [])
        for i in range(len(pixels) - 1):
            pt1 = (int(pixels[i][0]), int(pixels[i][1]))
            pt2 = (int(pixels[i + 1][0]), int(pixels[i + 1][1]))
            cv2.line(vis, pt1, pt2, (255, 0, 0), 1)

    # Draw nodes in red
    for node, data in G.nodes(data=True):
        pos = data.get('pos', (0, 0))
        cv2.circle(vis, (int(pos[0]), int(pos[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(filename, vis)
```

**Step 6: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py
git commit -m "feat: add path extraction and debug visualization"
```

---

## Task 7: Vectorizer - Combined Pipeline

**Files:**
- Modify: `modules/vectorizer.py`
- Modify: `tests/test_vectorizer.py`

**Step 1: Write failing test for raster_to_paths**

```python
# tests/test_vectorizer.py (append)

def test_raster_to_paths_full_pipeline():
    from modules.vectorizer import raster_to_paths
    from modules.utils import setup_output_dirs
    from PIL import Image

    setup_output_dirs()

    # Load cross fixture
    img = np.array(Image.open("tests/fixtures/simple_cross.png").convert("L"))
    binary = (img > 127).astype(np.uint8)

    paths = raster_to_paths(binary)

    # Should produce multiple paths
    assert len(paths) > 0
    # Each path should have coordinates
    assert all(len(p) >= 2 for p in paths)

    # Debug files should exist
    from pathlib import Path
    assert Path("output/debug/03_skeleton.png").exists()
    assert Path("output/debug/03_graph_nodes.png").exists()
    assert Path("output/debug/03_graph_pruned.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectorizer.py::test_raster_to_paths_full_pipeline -v`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
# modules/vectorizer.py (append)


def raster_to_paths(binary: np.ndarray) -> list[list[tuple[float, float]]]:
    """
    Full vectorization pipeline: binary → skeleton → graph → paths.

    Saves debug output at each stage to output/debug/.

    Args:
        binary: Binary image (0 and 1 values, foreground=1)

    Returns:
        List of paths, where each path is [(x, y), ...]
    """
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

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vectorizer.py::test_raster_to_paths_full_pipeline -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/vectorizer.py tests/test_vectorizer.py
git commit -m "feat: add raster_to_paths combining full pipeline"
```

---

## Task 8: Optimizer Module (vpype Integration)

**Files:**
- Create: `modules/optimizer.py`
- Create: `tests/test_optimizer.py`

**Step 1: Write failing test for optimize_paths**

```python
# tests/test_optimizer.py
from pathlib import Path


def test_optimize_paths_creates_document():
    from modules.optimizer import optimize_paths
    from modules.utils import setup_output_dirs

    setup_output_dirs()

    # Simple square path
    paths = [
        [(10, 10), (90, 10), (90, 90), (10, 90), (10, 10)]
    ]

    doc = optimize_paths(
        paths,
        width_mm=100,
        height_mm=100,
        source_width_px=100,
        source_height_px=100,
    )

    # Should return a vpype Document
    import vpype as vp
    assert isinstance(doc, vp.Document)

    # Debug files should exist
    assert Path("output/debug/03_paths.svg").exists()
    assert Path("output/debug/04_optimized.svg").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_optimizer.py::test_optimize_paths_creates_document -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# modules/optimizer.py
from pathlib import Path
import vpype as vp


def optimize_paths(
    paths: list[list[tuple[float, float]]],
    width_mm: float,
    height_mm: float,
    source_width_px: int,
    source_height_px: int,
) -> vp.Document:
    """
    Optimize paths for pen plotter efficiency.

    Args:
        paths: List of coordinate lists in pixel space
        width_mm: Target SVG width in mm
        height_mm: Target SVG height in mm
        source_width_px: Original image width in pixels
        source_height_px: Original image height in pixels

    Returns:
        vpype Document with optimized paths
    """
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_optimizer.py::test_optimize_paths_creates_document -v`
Expected: PASS

**Step 5: Write test for save_final_svg**

```python
# tests/test_optimizer.py (append)

def test_save_final_svg_writes_file():
    from modules.optimizer import optimize_paths, save_final_svg
    from modules.utils import setup_output_dirs

    setup_output_dirs()

    paths = [[(10, 10), (90, 10), (90, 90)]]

    doc = optimize_paths(paths, 100, 100, 100, 100)

    output_path = Path("output/test_final.svg")
    save_final_svg(doc, output_path, 100, 100)

    assert output_path.exists()
    content = output_path.read_text()
    assert "svg" in content
    assert "100mm" in content
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_optimizer.py::test_save_final_svg_writes_file -v`
Expected: FAIL with "ImportError"

**Step 7: Implement save_final_svg**

```python
# modules/optimizer.py (append)


def save_final_svg(
    doc: vp.Document,
    output_path: Path,
    width_mm: float,
    height_mm: float,
) -> None:
    """
    Save final SVG with correct dimensions.

    Args:
        doc: vpype Document
        output_path: Where to save the SVG
        width_mm: Page width in mm
        height_mm: Page height in mm
    """
    vp.write_svg(
        output_path,
        doc,
        page_size=(f"{width_mm}mm", f"{height_mm}mm"),
        center=True,
    )
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_optimizer.py -v`
Expected: PASS (both tests)

**Step 9: Commit**

```bash
git add modules/optimizer.py tests/test_optimizer.py
git commit -m "feat: add optimizer module with vpype integration"
```

---

## Task 9: Prompt Engineer Module

**Files:**
- Create: `modules/prompt_engineer.py`
- Create: `tests/test_prompt.py`

**Step 1: Write test for enhance_prompt (mocked)**

```python
# tests/test_prompt.py
from unittest.mock import patch, MagicMock


def test_enhance_prompt_calls_openrouter():
    from modules.utils import setup_output_dirs
    setup_output_dirs()

    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Enhanced prompt here"

    with patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_MODEL': 'test-model'
    }):
        with patch('modules.prompt_engineer.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response

            from modules.prompt_engineer import enhance_prompt
            result = enhance_prompt("a skull")

            assert result == "Enhanced prompt here"
            mock_client.assert_called_once()


def test_enhance_prompt_saves_debug():
    from modules.utils import setup_output_dirs
    from pathlib import Path
    setup_output_dirs()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Minimalistic line drawing of a skull"

    with patch.dict('os.environ', {
        'OPENROUTER_API_KEY': 'test-key',
        'OPENROUTER_MODEL': 'test-model'
    }):
        with patch('modules.prompt_engineer.OpenAI') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response

            from modules.prompt_engineer import enhance_prompt
            enhance_prompt("a skull")

            debug_file = Path("output/debug/01_prompt_enhanced.txt")
            assert debug_file.exists()
            content = debug_file.read_text()
            assert "a skull" in content
            assert "Minimalistic" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# modules/prompt_engineer.py
import os
from openai import OpenAI
from modules.utils import save_debug


SYSTEM_PROMPT = """You rewrite user prompts for Flux.2 image generation,
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

Output ONLY the rewritten prompt."""


def enhance_prompt(user_prompt: str) -> str:
    """
    Enhance a user prompt for Flux.2 line art generation.

    Args:
        user_prompt: Raw user input

    Returns:
        Enhanced prompt optimized for Flux.2 line art
    """
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

    save_debug(
        "01_prompt_enhanced.txt",
        f"Original: {user_prompt}\n\nEnhanced: {enhanced}"
    )

    return enhanced
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add modules/prompt_engineer.py tests/test_prompt.py
git commit -m "feat: add prompt engineer module for OpenRouter"
```

---

## Task 10: Raster Generator Module

**Files:**
- Create: `modules/raster_generator.py`
- Create: `tests/test_raster.py`

**Step 1: Write test for binary conversion (without GPU)**

```python
# tests/test_raster.py
import numpy as np
from PIL import Image
from pathlib import Path


def test_to_binary_converts_grayscale():
    from modules.raster_generator import to_binary
    from modules.utils import setup_output_dirs

    setup_output_dirs()

    # Create a grayscale image with clear dark/light regions
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[:, :50] = 30   # Dark left half
    gray[:, 50:] = 220  # Light right half

    img = Image.fromarray(gray)
    binary = to_binary(img)

    # Left half should be foreground (1), right half background (0)
    # (dark pixels become foreground in our convention)
    assert binary[:, :50].mean() > 0.9
    assert binary[:, 50:].mean() < 0.1


def test_to_binary_validates_content():
    from modules.raster_generator import to_binary
    from modules.utils import setup_output_dirs
    import pytest

    setup_output_dirs()

    # Blank white image
    white = Image.fromarray(np.full((100, 100), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="blank"):
        to_binary(white)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_raster.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation (binary conversion part)**

```python
# modules/raster_generator.py
import os
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from modules.utils import save_debug


def to_binary(image: Image.Image) -> np.ndarray:
    """
    Convert image to binary (threshold + validate).

    Args:
        image: PIL Image (any mode)

    Returns:
        Binary numpy array (0 and 1, foreground=1)

    Raises:
        ValueError: If image is blank or nearly blank
    """
    # Convert to grayscale
    gray = np.array(image.convert("L"))

    # Binary threshold (Otsu's method)
    thresh = threshold_otsu(gray)
    binary = (gray < thresh).astype(np.uint8)

    # Ensure foreground is minority (lines, not background)
    if np.mean(binary) > 0.5:
        binary = 1 - binary

    # Save debug
    Image.fromarray(binary * 255).save("output/debug/02_raster_binary.png")

    # Validate - check if image is not blank
    if np.sum(binary) < 0.01 * binary.size:
        raise ValueError("Generated image is blank or nearly blank")

    return binary
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_raster.py -v`
Expected: PASS (both tests)

**Step 5: Add generate_raster function (requires GPU, skip test)**

```python
# modules/raster_generator.py (add imports at top)
import torch
from diffusers import FluxPipeline


# modules/raster_generator.py (append)


def generate_raster(prompt: str) -> tuple[Image.Image, np.ndarray]:
    """
    Generate a raster image from prompt using Flux.2-dev.

    Args:
        prompt: Enhanced prompt for Flux.2

    Returns:
        Tuple of (raw PIL Image, binary numpy array)
    """
    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
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
    binary = to_binary(image)

    return image, binary
```

**Step 6: Commit**

```bash
git add modules/raster_generator.py tests/test_raster.py
git commit -m "feat: add raster generator module for Flux.2"
```

---

## Task 11: Main CLI Entry Point

**Files:**
- Create: `main.py`

**Step 1: Write main.py**

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
    parser.add_argument(
        "--width",
        type=float,
        default=420,
        help="Output width in mm (default: 420 for A3)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=297,
        help="Output height in mm (default: 297 for A3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: auto-timestamped)",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Skip LLM prompt enhancement, use raw prompt",
    )

    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = setup_output_dirs()
    stats = {"timestamp": timestamp, "stages": {}}

    # Stage 1: Prompt Enhancement
    print("[1/5] Enhancing prompt...")
    t0 = time.time()
    if args.skip_enhance:
        enhanced_prompt = args.prompt
        save_debug("01_prompt_enhanced.txt", f"Original (no enhancement): {args.prompt}")
    else:
        enhanced_prompt = enhance_prompt(args.prompt)
    stats["stages"]["prompt"] = {"time": time.time() - t0}
    print(f"       {enhanced_prompt[:80]}...")

    # Stage 2: Raster Generation
    print("[2/5] Generating raster...")
    t0 = time.time()
    raster, binary = generate_raster(enhanced_prompt)
    stats["stages"]["raster"] = {"time": time.time() - t0}
    print(f"       Shape: {binary.shape}")

    # Stage 3: Vectorization
    print("[3/5] Vectorizing...")
    t0 = time.time()
    paths = raster_to_paths(binary)
    stats["stages"]["vectorize"] = {
        "time": time.time() - t0,
        "path_count": len(paths),
        "total_points": sum(len(p) for p in paths),
    }
    print(f"       {len(paths)} paths extracted")

    # Stage 4: Optimization
    print("[4/5] Optimizing paths...")
    t0 = time.time()
    doc = optimize_paths(
        paths,
        args.width,
        args.height,
        binary.shape[1],
        binary.shape[0],
    )
    stats["stages"]["optimize"] = {"time": time.time() - t0}
    print("       Done")

    # Stage 5: Output
    output_name = args.output or f"output_{timestamp}.svg"
    output_path = output_dir / output_name
    save_final_svg(doc, output_path, args.width, args.height)

    # Save stats
    stats["total_time"] = sum(s["time"] for s in stats["stages"].values())
    save_debug("stats.json", json.dumps(stats, indent=2))

    print(f"[5/5] Saved: {output_path}")
    print(f"       Debug: output/debug/")
    print(f"       Time: {stats['total_time']:.1f}s")


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x main.py
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point"
```

---

## Task 12: Integration Test (Manual, requires GPU)

**Step 1: Create .env file on server**

```bash
cp .env.example .env
# Edit .env with actual API keys
```

**Step 2: Run full pipeline**

```bash
python main.py "a geometric skull"
```

**Step 3: Verify outputs**

Check:
- `output/output_*.svg` exists and opens in browser
- `output/debug/01_prompt_enhanced.txt` has enhanced prompt
- `output/debug/02_raster_raw.png` shows line art
- `output/debug/02_raster_binary.png` is clean binary
- `output/debug/03_skeleton.png` shows thin lines
- `output/debug/03_graph_nodes.png` shows red dots at junctions
- `output/debug/03_graph_pruned.png` shows cleaned graph
- `output/debug/03_paths.svg` has raw paths
- `output/debug/04_optimized.svg` has optimized paths
- `output/debug/stats.json` has timing info

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete txt2svg pipeline"
git push origin main
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 0 | Project Setup | pyproject.toml, .gitignore |
| 1 | Utils | modules/utils.py |
| 2 | Skeletonization | modules/vectorizer.py |
| 3 | Neighbor Counting | modules/vectorizer.py |
| 4 | Skeleton→Graph | modules/vectorizer.py |
| 5 | Spur Pruning | modules/vectorizer.py |
| 6 | Path Extraction | modules/vectorizer.py |
| 7 | Combined Pipeline | modules/vectorizer.py |
| 8 | Optimizer | modules/optimizer.py |
| 9 | Prompt Engineer | modules/prompt_engineer.py |
| 10 | Raster Generator | modules/raster_generator.py |
| 11 | CLI | main.py |
| 12 | Integration Test | (manual) |
