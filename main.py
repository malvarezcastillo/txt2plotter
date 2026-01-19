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
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )

    args = parser.parse_args()

    # Setup
    output_dir = setup_output_dirs()

    # Stage 1: Prompt Enhancement (once for all runs)
    t0 = time.time()
    if args.skip_enhance:
        enhanced_prompt = args.prompt
        save_debug("01_prompt_enhanced.txt", f"Original (no enhancement): {args.prompt}")
    else:
        enhanced_prompt = enhance_prompt(args.prompt)
    prompt_time = time.time() - t0
    print(f"[1/5] Prompt: {enhanced_prompt[:80]}...")

    # Generate n images
    for i in range(args.count):
        if args.count > 1:
            print(f"\n{'='*50}")
            print(f"Generating image {i + 1}/{args.count}")
            print(f"{'='*50}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats: dict = {"timestamp": timestamp, "run": i + 1, "stages": {}}
        stats["stages"]["prompt"] = {"time": prompt_time if i == 0 else 0}

        # Stage 2: Raster Generation
        t0 = time.time()
        # When using -n with seed, increment seed for each run
        run_seed = None
        if args.seed is not None:
            run_seed = args.seed + i
            print(f"      Using seed: {run_seed}")
        raster, binary = generate_raster(enhanced_prompt, seed=run_seed)
        stats["stages"]["raster"] = {"time": time.time() - t0, "seed": run_seed}
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
        if args.output and args.count == 1:
            output_name = args.output
        else:
            output_name = f"output_{timestamp}.svg"
        output_path = output_dir / output_name
        save_final_svg(doc, output_path, args.width, args.height)

        # Save stats
        stats["total_time"] = sum(s["time"] for s in stats["stages"].values())
        stats_filename = f"stats_{timestamp}.json" if args.count > 1 else "stats.json"
        save_debug(stats_filename, json.dumps(stats, indent=2))

        print(f"[5/5] Saved: {output_path}")
        print(f"      Debug files: output/debug/")
        print(f"      Total time: {stats['total_time']:.1f}s")


if __name__ == "__main__":
    main()
