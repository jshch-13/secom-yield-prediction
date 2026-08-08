"""Small shared helpers: reproducibility, I/O, and figure-saving utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs for reproducibility.

    Args:
        seed: Seed value applied to ``random`` and ``numpy.random``.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if it does not exist yet.

    Args:
        path: Directory path to create.

    Returns:
        The same path, for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts common NumPy scalar/array types."""

    def default(self, o: Any) -> Any:  # noqa: D102 - trivial override
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary as pretty-printed JSON, creating parent dirs as needed.

    Args:
        data: JSON-serializable dictionary (NumPy scalars/arrays are handled).
        path: Destination file path.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyJSONEncoder)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed dictionary contents.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_current_figure(path: Path, dpi: int = 150) -> None:
    """Save the current matplotlib figure to ``path`` and close it.

    Args:
        path: Destination PNG path. Parent directories are created if missing.
        dpi: Resolution used when saving.
    """
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
