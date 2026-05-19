default:
    @just --list

# Install core deps (fal.ai backend, no GPU needed)
install:
    pip install -e .

# Install with local 4-bit Flux backend (requires 24GB VRAM)
install-gpu:
    pip install -e ".[local-gpu]"

# Install dev deps (pytest)
install-dev:
    pip install -e ".[dev]"

# Run pytest
test:
    pytest

# Generate an SVG from a prompt (A3, fal.ai backend)
run PROMPT:
    python main.py "{{PROMPT}}"

# Generate with a fixed seed for reproducibility
run-seed PROMPT SEED:
    python main.py "{{PROMPT}}" --seed {{SEED}}

# Generate N variations of a prompt
run-many PROMPT N:
    python main.py "{{PROMPT}}" -n {{N}}

# Generate using the local GPU backend instead of fal.ai
run-local PROMPT:
    python main.py "{{PROMPT}}" --local-flux

# Run the batch file of example prompts
batch FILE="example_prompts.txt" N="1":
    python main.py --batch {{FILE}} -n {{N}}

# Remove all generated outputs
clean:
    rm -rf output/
