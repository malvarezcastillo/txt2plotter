"""Stage 3: Vectorization - skeleton to graph to paths."""

from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


def skeletonize_image(binary: np.ndarray, debug_dir: Path | None = None) -> np.ndarray:
    """Skeletonize a binary image using Lee's method.

    Args:
        binary: Binary image array (0 and 1 values).
        debug_dir: Directory to save debug files (None to skip debug output).

    Returns:
        Skeletonized image array.
    """
    # Lee's method produces smoother, better-connected skeletons
    skeleton = skeletonize(binary, method="lee")

    # Save debug
    if debug_dir:
        Image.fromarray((skeleton * 255).astype(np.uint8)).save(
            debug_dir / "03_skeleton.png"
        )

    return skeleton.astype(np.uint8)


def get_neighbors(y: int, x: int, skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Return coordinates of neighboring skeleton pixels (8-connected).

    Args:
        y: Row coordinate.
        x: Column coordinate.
        skeleton: Skeleton image array.

    Returns:
        List of (y, x) coordinates of neighboring skeleton pixels.
    """
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


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert a skeleton image to a networkx graph.

    Nodes are endpoints (1 neighbor) and junctions (3+ neighbors).
    Edges store the pixel path between nodes.

    Args:
        skeleton: Skeletonized binary image.

    Returns:
        NetworkX graph with nodes and edges.
    """
    # Step 1: Count neighbors for each pixel using convolution
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    neighbor_count = ndimage.convolve(skeleton, kernel, mode="constant")
    neighbor_count = neighbor_count * skeleton  # Only count skeleton pixels

    # Step 2: Identify node pixels (endpoints + junctions)
    endpoints = (neighbor_count == 1) & (skeleton == 1)  # Dead ends
    junctions = (neighbor_count >= 3) & (skeleton == 1)  # Intersections
    node_mask = endpoints | junctions

    # Step 3: Label each node with unique ID
    node_coords = np.argwhere(node_mask)  # [(y, x), ...]
    coord_to_node = {tuple(c): i for i, c in enumerate(node_coords)}

    # Step 4: Create graph, add nodes
    G = nx.Graph()
    for i, (y, x) in enumerate(node_coords):
        G.add_node(i, pos=(int(x), int(y)))  # Note: (x, y) for SVG coords

    # Step 5: Trace edges between nodes
    visited_edges: set[frozenset] = set()

    for start_idx, (sy, sx) in enumerate(node_coords):
        for ny, nx_ in get_neighbors(sy, sx, skeleton):
            edge_key = frozenset([(sy, sx), (ny, nx_)])
            if edge_key in visited_edges:
                continue

            # Trace path until we hit another node
            path = [(int(sx), int(sy))]  # Store as (x, y)
            prev, curr = (sy, sx), (ny, nx_)

            while True:
                path.append((int(curr[1]), int(curr[0])))  # (x, y)
                visited_edges.add(frozenset([prev, curr]))

                if node_mask[curr[0], curr[1]]:
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


def prune_spurs(G: nx.Graph, min_length: int = 10) -> nx.Graph:
    """Remove leaf edges shorter than min_length pixels.

    Args:
        G: Input graph.
        min_length: Minimum edge length to keep.

    Returns:
        Pruned graph.
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
            pixels = edge[2].get("pixels", [])

            if len(pixels) < min_length:
                G.remove_node(leaf)
                pruned = True  # May expose new leaves, iterate

    # Remove isolated nodes (degree 0)
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])

    return G


def extract_paths(G: nx.Graph) -> list[list[tuple[float, float]]]:
    """Extract all edge pixel chains as coordinate lists.

    Args:
        G: Graph with pixel paths on edges.

    Returns:
        List of paths, each path is a list of (x, y) coordinates.
    """
    paths = []

    for u, v, data in G.edges(data=True):
        pixels = data.get("pixels", [])
        if len(pixels) >= 2:
            paths.append([(float(x), float(y)) for x, y in pixels])

    return paths


def save_graph_debug(
    skeleton: np.ndarray, G: nx.Graph, filename: str, debug_dir: Path | None = None
) -> None:
    """Save visualization with nodes (red) and edges (blue).

    Args:
        skeleton: Skeleton image for background.
        G: Graph to visualize.
        filename: Output filename.
        debug_dir: Directory to save debug files (None to skip debug output).
    """
    if not debug_dir:
        return

    vis = cv2.cvtColor((skeleton * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw edges in blue
    for u, v, data in G.edges(data=True):
        pixels = data.get("pixels", [])
        for i in range(len(pixels) - 1):
            pt1 = (int(pixels[i][0]), int(pixels[i][1]))
            pt2 = (int(pixels[i + 1][0]), int(pixels[i + 1][1]))
            cv2.line(vis, pt1, pt2, (255, 0, 0), 1)

    # Draw nodes in red
    for node, data in G.nodes(data=True):
        pos = data.get("pos", (0, 0))
        cv2.circle(vis, (int(pos[0]), int(pos[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(str(debug_dir / filename), vis)


def raster_to_paths(
    binary: np.ndarray, debug_dir: Path | None = None
) -> list[list[tuple[float, float]]]:
    """Full vectorization pipeline: binary -> skeleton -> graph -> paths.

    Args:
        binary: Binary image array.
        debug_dir: Directory to save debug files (None to skip debug output).

    Returns:
        List of paths for SVG conversion.
    """
    # Skeletonize
    skeleton = skeletonize_image(binary, debug_dir=debug_dir)

    # Build graph
    G = skeleton_to_graph(skeleton)
    save_graph_debug(skeleton, G, "03_graph_nodes.png", debug_dir=debug_dir)

    # Prune spurs
    G = prune_spurs(G, min_length=10)
    save_graph_debug(skeleton, G, "03_graph_pruned.png", debug_dir=debug_dir)

    # Extract paths
    paths = extract_paths(G)

    return paths
