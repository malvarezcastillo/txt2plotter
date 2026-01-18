#!/usr/bin/env python3
"""Text-to-SVG pipeline for AxiDraw pen plotters."""

import argparse
import json
import time
from datetime import datetime

from dotenv import load_dotenv

from modules.optimizer import optimize_paths, save_final_svg
from modules.prompt_engineer import enhance_prompt
from modules.raster_generator import generate_raster
from modules.utils import save_debug, setup_output_dirs
from modules.vectorizer import raster_to_paths


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
    stats: dict = {"timestamp": timestamp, "stages": {}}

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
        paths, args.width, args.height, binary.shape[1], binary.shape[0]
    )
    stats["stages"]["optimize"] = {"time": time.time() - t0}
    print("[4/5] Optimized paths")

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
