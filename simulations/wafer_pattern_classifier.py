"""Rule-based spatial pattern classifier for the synthetic wafer-map demo.

**This is a demonstration classifier, not a validated production model.**
It uses simple geometric heuristics (fail rate, radial concentration,
linear-fit spread) to guess which synthetic pattern
(``EDGE_RING``/``CENTER``/``SCRATCH``/``RANDOM``/``CLEAN``) a wafer's die map
resembles. It has not been evaluated on real inspection data and must never
be described as validated in a manufacturing environment.

Run as a script to classify every wafer in the synthetic dataset:

    python simulations/wafer_pattern_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulations.generate_synthetic_wafer_data import DEFAULT_OUTPUT_PATH, generate_dataset  # noqa: E402

METHOD_LABEL = "Rule-based demonstration classifier"

CLEAN_FAIL_RATE_THRESHOLD = 0.05
# Note: for uniformly RANDOM failures over a disk, E[radius/R] = 2/3 by
# geometry (area grows with radius), not 0.5 -- thresholds are calibrated
# against that baseline rather than a naive midpoint.
EDGE_RADIUS_NORM_THRESHOLD = 0.80
CENTER_RADIUS_NORM_THRESHOLD = 0.45
SCRATCH_ASPECT_RATIO_THRESHOLD = 0.55
MIN_FAIL_DIES_FOR_SPATIAL_CALL = 5


def _require_synthetic(df: pd.DataFrame) -> None:
    """Raise if any row is not explicitly flagged as synthetic demo data.

    Args:
        df: Wafer die-level DataFrame to validate.

    Raises:
        ValueError: If ``synthetic_data_flag`` is missing or not all True.
    """
    if "synthetic_data_flag" not in df.columns or not bool(df["synthetic_data_flag"].all()):
        raise ValueError(
            "wafer_pattern_classifier only accepts synthetic_data_flag=True rows; "
            "it must never be run against real SECOM-derived or production data."
        )


def _line_fit_aspect_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Ratio of spread perpendicular to a fitted line vs. spread along it.

    A low ratio means points cluster tightly around a straight line (as a
    SCRATCH pattern would), while a ratio near 1 means the spread is roughly
    isotropic (consistent with RANDOM).

    Args:
        x: X coordinates of the points being fit.
        y: Y coordinates of the points being fit.

    Returns:
        ``std_perpendicular / std_along_axis`` from a PCA of the point cloud.
    """
    points = np.column_stack([x - x.mean(), y - y.mean()])
    cov = np.cov(points.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    minor, major = np.sort(eigvals)
    return float(np.sqrt(minor) / np.sqrt(major))


def classify_wafer(wafer_df: pd.DataFrame, wafer_radius_mm: float | None = None) -> dict[str, Any]:
    """Classify one wafer's die map into a synthetic defect-pattern category.

    Args:
        wafer_df: Rows for exactly one ``(lot_id, wafer_id)`` pair.
        wafer_radius_mm: Wafer radius used to normalize radius; defaults to
            the max observed radius in ``wafer_df`` if not given.

    Returns:
        Dictionary with ``lot_id``, ``wafer_id``, ``true_pattern`` (if
        present), ``predicted_pattern``, ``fail_rate``, and ``method``.
    """
    _require_synthetic(wafer_df)

    lot_id = wafer_df["lot_id"].iloc[0]
    wafer_id = wafer_df["wafer_id"].iloc[0]
    true_pattern = wafer_df["defect_pattern"].iloc[0] if "defect_pattern" in wafer_df else None

    fail_df = wafer_df[wafer_df["status"] == "FAIL"]
    fail_rate = len(fail_df) / len(wafer_df) if len(wafer_df) else 0.0
    radius_scale = wafer_radius_mm or wafer_df["radius"].max()

    if fail_rate < CLEAN_FAIL_RATE_THRESHOLD or len(fail_df) < MIN_FAIL_DIES_FOR_SPATIAL_CALL:
        predicted = "CLEAN"
    else:
        mean_radius_norm = float((fail_df["radius"] / radius_scale).mean())
        aspect_ratio = _line_fit_aspect_ratio(
            fail_df["die_x"].to_numpy(), fail_df["die_y"].to_numpy()
        )
        if mean_radius_norm >= EDGE_RADIUS_NORM_THRESHOLD:
            predicted = "EDGE_RING"
        elif mean_radius_norm <= CENTER_RADIUS_NORM_THRESHOLD:
            predicted = "CENTER"
        elif aspect_ratio <= SCRATCH_ASPECT_RATIO_THRESHOLD:
            predicted = "SCRATCH"
        else:
            predicted = "RANDOM"

    return {
        "lot_id": lot_id,
        "wafer_id": wafer_id,
        "true_pattern": true_pattern,
        "predicted_pattern": predicted,
        "fail_rate": round(fail_rate, 4),
        "method": METHOD_LABEL,
    }


def classify_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run :func:`classify_wafer` over every ``(lot_id, wafer_id)`` group in ``df``.

    Args:
        df: Full synthetic wafer dataset (multiple wafers).

    Returns:
        DataFrame with one row of classification results per wafer.
    """
    _require_synthetic(df)
    results = [
        classify_wafer(group)
        for _, group in df.groupby(["lot_id", "wafer_id"], sort=False)
    ]
    return pd.DataFrame(results)


if __name__ == "__main__":
    data = (
        pd.read_csv(DEFAULT_OUTPUT_PATH) if DEFAULT_OUTPUT_PATH.exists() else generate_dataset()
    )
    results_df = classify_dataset(data)
    accuracy = float((results_df["predicted_pattern"] == results_df["true_pattern"]).mean())
    print(f"[SYNTHETIC DEMO] {METHOD_LABEL} -- agreement with generator label: {accuracy:.2%} "
          f"(this is a heuristic sanity check on synthetic data, not a validated model metric)")
    print(results_df.to_string(index=False))
