"""Shared utilities for the txt2svg pipeline."""

from pathlib import Path

# Default paths
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def setup_output_dirs() -> Path:
    """Create base output directory if it doesn't exist.

    Returns:
        Path to the output directory.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def create_run_dir(timestamp: str, parent: Path | None = None) -> tuple[Path, Path]:
    """Create a directory for a single run with its debug subdirectory.

    Args:
        timestamp: Timestamp string to use as directory name.
        parent: Parent directory (defaults to OUTPUT_DIR).

    Returns:
        Tuple of (run_dir, debug_dir) paths.
    """
    base = parent or OUTPUT_DIR
    run_dir = base / f"run_{timestamp}"
    debug_dir = run_dir / "debug"
    run_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    return run_dir, debug_dir


def save_debug(filename: str, content: str | bytes, debug_dir: Path) -> Path:
    """Save debug content to the specified debug directory.

    Args:
        filename: Name of the file to save.
        content: Content to write (str or bytes).
        debug_dir: Path to the debug directory.

    Returns:
        Path to the saved file.
    """
    filepath = debug_dir / filename

    if isinstance(content, bytes):
        filepath.write_bytes(content)
    else:
        filepath.write_text(content)

    return filepath
