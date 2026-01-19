"""Stage 4: Path optimization using vpype."""

from pathlib import Path

import numpy as np
import vpype as vp
from shapely.geometry import LineString

from .utils import DEBUG_DIR


def _write_svg_to_file(filepath: Path, doc: vp.Document, **kwargs) -> None:
    """Write vpype document to SVG file.

    Args:
        filepath: Path to output file.
        doc: vpype Document to write.
        **kwargs: Additional arguments for write_svg.
    """
    with open(filepath, "w") as f:
        vp.write_svg(f, doc, **kwargs)


def _simplify_line(line: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a line using Douglas-Peucker algorithm via shapely.

    Args:
        line: Complex array of points (x + yj).
        tolerance: Simplification tolerance.

    Returns:
        Simplified line as complex array.
    """
    if len(line) < 3:
        return line

    # Convert to coordinate pairs for shapely
    coords = [(p.real, p.imag) for p in line]
    ls = LineString(coords)
    simplified = ls.simplify(tolerance, preserve_topology=True)

    # Convert back to complex array
    return np.array([complex(x, y) for x, y in simplified.coords])


def _sort_lines_greedy(lines: list[np.ndarray]) -> list[np.ndarray]:
    """Sort lines to minimize pen-up travel using greedy nearest-neighbor.

    Args:
        lines: List of lines (complex arrays).

    Returns:
        Sorted list of lines.
    """
    if len(lines) <= 1:
        return lines

    sorted_lines = []
    remaining = list(range(len(lines)))
    current_pos = complex(0, 0)

    while remaining:
        # Find nearest line start/end
        best_idx = None
        best_dist = float("inf")
        best_reverse = False

        for idx in remaining:
            line = lines[idx]
            start_dist = abs(line[0] - current_pos)
            end_dist = abs(line[-1] - current_pos)

            if start_dist < best_dist:
                best_dist = start_dist
                best_idx = idx
                best_reverse = False

            if end_dist < best_dist:
                best_dist = end_dist
                best_idx = idx
                best_reverse = True

        # Add the best line
        line = lines[best_idx]
        if best_reverse:
            line = line[::-1]

        sorted_lines.append(line)
        current_pos = line[-1]
        remaining.remove(best_idx)

    return sorted_lines


def optimize_paths(
    paths: list[list[tuple[float, float]]],
    width_mm: float,
    height_mm: float,
    source_width_px: int,
    source_height_px: int,
) -> vp.Document:
    """Optimize paths using vpype for pen plotter output.

    Operations:
    - linemerge: Connect nearby endpoints into continuous strokes
    - linesimplify: Reduce vertices, smooth pixel jitter
    - linesort: Greedy nearest-neighbor minimizes pen-up travel time
    - reloop: Align loop start/end for clean closure

    Args:
        paths: List of paths, each path is a list of (x, y) pixel coordinates.
        width_mm: Target width in millimeters.
        height_mm: Target height in millimeters.
        source_width_px: Source image width in pixels.
        source_height_px: Source image height in pixels.

    Returns:
        Optimized vpype Document.
    """
    # Create vpype document
    doc = vp.Document()
    lc = vp.LineCollection()

    # Scale factor: pixels -> mm
    scale_x = width_mm / source_width_px
    scale_y = height_mm / source_height_px

    # Convert paths to vpype lines (complex numbers: x + yj)
    for path in paths:
        if len(path) < 2:
            continue
        line = np.array([complex(x * scale_x, y * scale_y) for x, y in path])
        lc.append(line)

    doc.add(lc, layer_id=1)

    # Save pre-optimization debug
    _write_svg_to_file(DEBUG_DIR / "03_paths.svg", doc)

    # Optimization pipeline
    # 1. Merge nearby endpoints (tolerance in mm, convert from vpype units)
    merge_tolerance = 0.1  # mm
    lc = doc.layers[1]
    lc.merge(tolerance=merge_tolerance)

    # 2. Simplify lines to reduce vertices
    simplify_tolerance = 0.05  # mm
    simplified_lines = []
    for line in lc.lines:
        simplified = _simplify_line(line, simplify_tolerance)
        if len(simplified) >= 2:
            simplified_lines.append(simplified)

    # 3. Sort lines to minimize pen-up travel
    sorted_lines = _sort_lines_greedy(simplified_lines)

    # 4. Reloop closed paths
    final_lc = vp.LineCollection()
    for line in sorted_lines:
        final_lc.append(line)
    final_lc.reloop(tolerance=0.1)

    # Create final document
    doc = vp.Document()
    doc.add(final_lc, layer_id=1)

    # Save post-optimization debug
    _write_svg_to_file(DEBUG_DIR / "04_optimized.svg", doc)

    return doc


def save_final_svg(
    doc: vp.Document,
    output_path: Path,
    width_mm: float,
    height_mm: float,
    prompt: str | None = None,
    enhanced_prompt: str | None = None,
) -> None:
    """Save the final SVG with proper page dimensions.

    Args:
        doc: vpype Document to save.
        output_path: Output file path.
        width_mm: Page width in millimeters.
        height_mm: Page height in millimeters.
        prompt: Original user prompt to embed as comment.
        enhanced_prompt: Enhanced prompt to embed as comment.
    """
    # Convert mm to pixels (96 DPI, 1 inch = 25.4mm)
    px_per_mm = 96.0 / 25.4
    width_px = width_mm * px_per_mm
    height_px = height_mm * px_per_mm

    _write_svg_to_file(
        output_path,
        doc,
        page_size=(width_px, height_px),
        center=True,
    )

    # Post-process SVG for better Inkscape compatibility
    svg_content = output_path.read_text()
    # Add explicit fill and stroke to each polyline so copy-paste works in Inkscape
    svg_content = svg_content.replace("<polyline ", '<polyline fill="none" stroke="#000000" ')

    # Embed prompts as XML comments
    if prompt or enhanced_prompt:
        comment_lines = ["", "  Generated by txt2plotter"]
        if prompt:
            comment_lines.append(f"  Prompt: {prompt}")
        if enhanced_prompt and enhanced_prompt != prompt:
            comment_lines.append(f"  Enhanced: {enhanced_prompt}")
        comment_lines.append("")
        comment = "<!--" + "\n".join(comment_lines) + "-->\n"
        # Insert after XML declaration or at start
        if svg_content.startswith("<?xml"):
            insert_pos = svg_content.index("?>") + 2
            svg_content = svg_content[:insert_pos] + "\n" + comment + svg_content[insert_pos:]
        else:
            svg_content = comment + svg_content

    output_path.write_text(svg_content)
