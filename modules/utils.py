"""Shared utilities for the txt2svg pipeline."""

from pathlib import Path

# Default paths
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEBUG_DIR = OUTPUT_DIR / "debug"


def setup_output_dirs() -> Path:
    """Create output directories if they don't exist.

    Returns:
        Path to the output directory.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def save_debug(filename: str, content: str | bytes) -> Path:
    """Save debug content to the debug directory.

    Args:
        filename: Name of the file to save.
        content: Content to write (str or bytes).

    Returns:
        Path to the saved file.
    """
    filepath = DEBUG_DIR / filename

    if isinstance(content, bytes):
        filepath.write_bytes(content)
    else:
        filepath.write_text(content)

    return filepath
