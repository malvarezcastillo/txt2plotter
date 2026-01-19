#!/usr/bin/env python3
"""Text-to-SVG pipeline for AxiDraw pen plotters."""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from modules.optimizer import optimize_paths, save_final_svg
from modules.prompt_engineer import enhance_prompt
from modules.raster_generator import generate_raster
from modules.utils import save_debug, setup_output_dirs
from modules.vectorizer import raster_to_paths


def slugify(text: str, max_length: int = 40) -> str:
    """Convert text to a filesystem-safe slug."""
    # Take first part, lowercase, replace non-alnum with underscore
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    slug = slug.strip("_")[:max_length].rstrip("_")
    return slug or "output"


def parse_batch_file(filepath: Path) -> list[str]:
    """Parse a batch file containing prompts.

    Supports:
    - One prompt per line
    - Lines starting with # are comments
    - Empty lines are skipped
    - Quoted strings (extracts content between quotes)
    """
    prompts = []
    content = filepath.read_text()

    for line in content.splitlines():
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Extract quoted string if present
        match = re.search(r'"([^"]+)"', line)
        if match:
            prompts.append(match.group(1))
        else:
            prompts.append(line)

    return prompts


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate plotter-ready SVG from text prompt"
    )
    parser.add_argument("prompt", nargs="?", help="Text description of desired image")
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
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to file containing prompts (one per line)",
    )

    args = parser.parse_args()

    # Validate args
    if not args.prompt and not args.batch:
        parser.error("Either prompt or --batch is required")

    # Collect prompts
    if args.batch:
        prompts = parse_batch_file(Path(args.batch))
        print(f"Loaded {len(prompts)} prompts from {args.batch}")
    else:
        prompts = [args.prompt]

    # Setup
    output_dir = setup_output_dirs()
    total_images = len(prompts) * args.count
    image_num = 0
    seed_offset = 0

    # Process each prompt
    for prompt_idx, raw_prompt in enumerate(prompts):
        if len(prompts) > 1:
            print(f"\n{'#'*60}")
            print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {raw_prompt[:60]}...")
            print(f"{'#'*60}")

        # Stage 1: Prompt Enhancement (once per prompt)
        t0 = time.time()
        if args.skip_enhance:
            enhanced_prompt = raw_prompt
            save_debug("01_prompt_enhanced.txt", f"Original (no enhancement): {raw_prompt}")
        else:
            enhanced_prompt = enhance_prompt(raw_prompt)
        prompt_time = time.time() - t0
        print(f"[1/5] Prompt: {enhanced_prompt[:80]}...")

        # Create prompt-specific output directory for batch mode
        if args.batch:
            prompt_slug = slugify(raw_prompt)
            prompt_dir = output_dir / prompt_slug
            prompt_dir.mkdir(exist_ok=True)
        else:
            prompt_dir = output_dir

        # Generate n images for this prompt
        for i in range(args.count):
            image_num += 1
            if args.count > 1:
                print(f"\n{'='*50}")
                print(f"Generating image {i + 1}/{args.count} [{image_num}/{total_images} total]")
                print(f"{'='*50}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats: dict = {"timestamp": timestamp, "prompt": raw_prompt, "run": i + 1, "stages": {}}
            stats["stages"]["prompt"] = {"time": prompt_time if i == 0 else 0}

            # Stage 2: Raster Generation
            t0 = time.time()
            # When using seed, increment for each image globally
            run_seed = None
            if args.seed is not None:
                run_seed = args.seed + seed_offset
                seed_offset += 1
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
            if args.output and args.count == 1 and len(prompts) == 1:
                output_name = args.output
            else:
                output_name = f"{timestamp}.svg"
            output_path = prompt_dir / output_name
            save_final_svg(doc, output_path, args.width, args.height, raw_prompt, enhanced_prompt)

            # Save stats
            stats["total_time"] = sum(s["time"] for s in stats["stages"].values())
            stats_filename = f"stats_{timestamp}.json"
            save_debug(stats_filename, json.dumps(stats, indent=2))

            print(f"[5/5] Saved: {output_path}")
            print(f"      Total time: {stats['total_time']:.1f}s")

    print(f"\n{'='*60}")
    print(f"Done! Generated {image_num} images.")
    if args.batch:
        print(f"Output organized in: {output_dir}/<prompt_slug>/")


if __name__ == "__main__":
    main()
