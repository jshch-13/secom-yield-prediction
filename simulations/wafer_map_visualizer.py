"""Render synthetic wafer die maps as PNG images (Matplotlib).

**Synthetic demo only.** Every figure produced here is titled with the
``[SYNTHETIC DEMO]`` marker and a footer disclaimer, since the underlying
data comes from ``simulations/generate_synthetic_wafer_data.py`` and has no
relationship to the SECOM dataset or any real Fab.

Run as a script to render one representative wafer map per defect pattern:

    python simulations/wafer_map_visualizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulations.generate_synthetic_wafer_data import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    WAFER_RADIUS_MM,
    generate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_SIMULATIONS_DIR = PROJECT_ROOT / "outputs" / "simulations"

DISCLAIMER = "Synthetic data - Not derived from SECOM or production Fab data"


def build_wafer_figure(
    wafer_df: pd.DataFrame, wafer_radius_mm: float = WAFER_RADIUS_MM
) -> plt.Figure:
    """Build a single wafer die-map figure (PASS/FAIL scatter with boundary + notch).

    Shared by :func:`plot_wafer_map` (single PNG) and
    ``simulations/batch_wafer_report.py`` (multi-page PDF) so both draw the
    exact same figure.

    Args:
        wafer_df: Rows for exactly one ``(lot_id, wafer_id)`` pair, with
            ``die_x``, ``die_y``, ``status``, ``defect_pattern`` columns.
        wafer_radius_mm: Wafer radius, used to draw the boundary circle.

    Returns:
        The Matplotlib figure (caller is responsible for saving/closing it).

    Raises:
        ValueError: If any row is missing the ``synthetic_data_flag`` marker.
    """
    if not wafer_df.get("synthetic_data_flag", pd.Series([False])).all():
        raise ValueError(
            "build_wafer_figure() only accepts synthetic_data_flag=True rows; "
            "this visualizer must never be pointed at real SECOM-derived data."
        )

    pattern = wafer_df["defect_pattern"].iloc[0]
    lot_id = wafer_df["lot_id"].iloc[0]
    wafer_id = wafer_df["wafer_id"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    boundary = patches.Circle(
        (0, 0), wafer_radius_mm, fill=False, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(boundary)

    # Notch marker at the conventional 6 o'clock (270 deg / -90 deg) position.
    notch_x, notch_y = 0.0, -wafer_radius_mm
    ax.plot(notch_x, notch_y, marker="v", markersize=10, color="black", zorder=5)

    passed = wafer_df[wafer_df["status"] == "PASS"]
    failed = wafer_df[wafer_df["status"] == "FAIL"]
    ax.scatter(passed["die_x"], passed["die_y"], s=14, c="#55A868", label="PASS", alpha=0.8)
    ax.scatter(failed["die_x"], failed["die_y"], s=14, c="#C44E52", label="FAIL", alpha=0.9)

    ax.set_xlim(-wafer_radius_mm * 1.1, wafer_radius_mm * 1.1)
    ax.set_ylim(-wafer_radius_mm * 1.1, wafer_radius_mm * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("die_x (mm)")
    ax.set_ylabel("die_y (mm)")
    ax.set_title(f"[SYNTHETIC DEMO] Wafer Map - {pattern}\n{lot_id} / {wafer_id}")
    ax.legend(loc="upper right")
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=8, color="#666666", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def plot_wafer_map(
    wafer_df: pd.DataFrame,
    output_path: Path,
    wafer_radius_mm: float = WAFER_RADIUS_MM,
) -> Path:
    """Render a single wafer's die map to a PNG file.

    Args:
        wafer_df: Rows for exactly one ``(lot_id, wafer_id)`` pair.
        output_path: Destination PNG path.
        wafer_radius_mm: Wafer radius, used to draw the boundary circle.

    Returns:
        The ``output_path`` the figure was saved to.
    """
    fig = build_wafer_figure(wafer_df, wafer_radius_mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_one_map_per_pattern(
    df: pd.DataFrame | None = None, output_dir: Path = OUTPUTS_SIMULATIONS_DIR
) -> list[Path]:
    """Render one representative wafer map PNG per defect pattern.

    Args:
        df: Synthetic wafer dataset (loads from ``DEFAULT_OUTPUT_PATH`` or
            regenerates it in-memory if ``None``).
        output_dir: Directory to write ``wafer_map_{pattern}.png`` files into.

    Returns:
        List of saved PNG paths.
    """
    if df is None:
        df = (
            pd.read_csv(DEFAULT_OUTPUT_PATH)
            if DEFAULT_OUTPUT_PATH.exists()
            else generate_dataset()
        )

    saved_paths = []
    for pattern, group in df.groupby("defect_pattern"):
        first_wafer_id = group["wafer_id"].iloc[0]
        wafer_df = group[group["wafer_id"] == first_wafer_id]
        path = output_dir / f"wafer_map_{pattern}.png"
        saved_paths.append(plot_wafer_map(wafer_df, path))
    return saved_paths


if __name__ == "__main__":
    paths = render_one_map_per_pattern()
    for p in paths:
        print(f"[SYNTHETIC DEMO] wrote {p}")
