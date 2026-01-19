# txt2plotter

Convert text prompts to pen-plotter-ready SVG files using AI image generation and centerline vectorization.

## Pipeline

1. **Prompt Enhancement** - LLM rewrites your prompt for optimal line art generation
2. **Raster Generation** - Flux.2-dev generates a high-contrast line art image
3. **Vectorization** - Skeletonization and graph extraction produce clean paths
4. **Optimization** - Paths are merged, simplified, and sorted for efficient plotting
5. **Output** - Plotter-ready SVG with configurable dimensions

## Examples

| | | |
|:---:|:---:|:---:|
| ![fox](examples/fox.svg) | ![cat](examples/cat_silhouette.svg) | ![hummingbird](examples/hummingbird.svg) |
| *"Fox sitting, geometric style"* | *"Sitting cat silhouette"* | *"Hummingbird in flight, art nouveau"* |
| ![jellyfish](examples/jellyfish.svg) | ![monstera](examples/monstera.svg) | ![coffee](examples/coffee_cup.svg) |
| *"Jellyfish with flowing tentacles"* | *"Monstera leaf with holes"* | *"Steaming coffee cup"* |
| ![mountain](examples/mountain_pines.svg) | ![waves](examples/waves.svg) | ![leaf](examples/leaf.svg) |
| *"Mountain peaks with pine trees"* | *"Ocean wave pattern, Japanese style"* | *"Leaf outline with vein pattern"* |

### Plotted

| | |
|:---:|:---:|
| ![plotter overview](examples/plotted_overview.jpg) | ![cat closeup](examples/plotted_cat_closeup.jpg) |
| *Multiple designs on the plotter* | *Cat silhouette closeup* |

## Requirements

- Python 3.10+
- NVIDIA GPU with 24GB VRAM (RTX 3090/4090)
- CUDA 12.x
- [OpenRouter API key](https://openrouter.ai/) for prompt enhancement
- [HuggingFace token](https://huggingface.co/settings/tokens) with access to Flux.2-dev

## Installation

```bash
git clone https://github.com/malvarezcastillo/txt2plotter.git
cd txt2plotter

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

## Usage

```bash
# Basic usage (A3 size)
python main.py "a geometric skull"

# Custom dimensions (A4)
python main.py "circuit board pattern" --width 297 --height 210

# Generate multiple variations
python main.py "mountain landscape" -n 5

# Reproducible generation with seed
python main.py "geometric pattern" --seed 42

# Generate 3 reproducible variations (uses seed, seed+1, seed+2)
python main.py "geometric pattern" -n 3 --seed 42

# Skip prompt enhancement
python main.py "minimalistic line drawing of a cat" --skip-enhance

# Batch mode: process multiple prompts from file
python main.py --batch prompts.txt -n 10
```

### Batch File Format

Create a `prompts.txt` file (one prompt per line, supports comments):

```
# My prompts
"Minimalistic isometric impossible cube, thick black lines, technical drawing."
"Single continuous line drawing of a greyhound, Picasso style."

# This is also valid (no quotes)
A geometric skull with clean vector lines
```

Output is organized by prompt: `output/<prompt_slug>/`

## Output

- `output/*.svg` - Final plotter-ready SVGs
- `output/debug/` - Intermediate files for debugging:
  - `01_prompt_enhanced.txt` - Enhanced prompt
  - `02_raster_raw.png` - Generated image
  - `02_raster_binary.png` - Thresholded binary
  - `03_skeleton.png` - Skeletonized paths
  - `03_graph_*.png` - Graph visualization
  - `03_paths.svg` - Raw paths
  - `04_optimized.svg` - After optimization

## License

MIT
