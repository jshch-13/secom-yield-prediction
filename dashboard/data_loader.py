"""Load the CSV artifacts produced by ``src/train.py`` for the Dash dashboard.

The dashboard is a read-only viewer over ``data/processed/`` -- it never
recomputes model predictions itself, only visualizes what the training
pipeline already produced. This keeps the numbers shown here identical to
the ones in ``outputs/metrics/`` and the project README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_CSV = DATA_PROCESSED_DIR / "secom_spotfire_master.csv"
FEATURE_SUMMARY_CSV = DATA_PROCESSED_DIR / "secom_feature_summary.csv"
MODEL_PERFORMANCE_CSV = DATA_PROCESSED_DIR / "secom_model_performance.csv"

MISSING_DATA_MESSAGE = (
    "No processed data found under data/processed/. Run `python src/train.py` "
    "(with real SECOM data placed under data/raw/, see data/README.md) to "
    "generate secom_spotfire_master.csv, secom_feature_summary.csv, and "
    "secom_model_performance.csv before starting this dashboard."
)


class DashboardDataError(RuntimeError):
    """Raised when a required processed-data CSV is missing."""


def data_is_available() -> bool:
    """Check whether all three required processed CSVs exist.

    Returns:
        True if the master table, feature summary, and model performance
        CSVs all exist under ``data/processed/``.
    """
    return MASTER_CSV.exists() and FEATURE_SUMMARY_CSV.exists() and MODEL_PERFORMANCE_CSV.exists()


def load_master_table() -> pd.DataFrame:
    """Load ``secom_spotfire_master.csv`` (row-level predictions + top features).

    Returns:
        The master table, with ``Time`` parsed as datetime.

    Raises:
        DashboardDataError: If the file does not exist.
    """
    if not MASTER_CSV.exists():
        raise DashboardDataError(MISSING_DATA_MESSAGE)
    df = pd.read_csv(MASTER_CSV)
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df


def load_feature_summary() -> pd.DataFrame:
    """Load ``secom_feature_summary.csv`` (per-feature missing ratio/variance/importance).

    Returns:
        The feature summary table.

    Raises:
        DashboardDataError: If the file does not exist.
    """
    if not FEATURE_SUMMARY_CSV.exists():
        raise DashboardDataError(MISSING_DATA_MESSAGE)
    return pd.read_csv(FEATURE_SUMMARY_CSV)


def load_model_performance() -> pd.DataFrame:
    """Load ``secom_model_performance.csv`` (candidate-model comparison table).

    Returns:
        The model comparison table.

    Raises:
        DashboardDataError: If the file does not exist.
    """
    if not MODEL_PERFORMANCE_CSV.exists():
        raise DashboardDataError(MISSING_DATA_MESSAGE)
    return pd.read_csv(MODEL_PERFORMANCE_CSV)


def get_feature_columns(master_df: pd.DataFrame) -> list[str]:
    """Return the anonymized sensor feature columns present in the master table.

    Args:
        master_df: Output of :func:`load_master_table`.

    Returns:
        Column names starting with ``feature_``, in their existing order
        (already ranked by importance when written by ``src/train.py``).
    """
    return [c for c in master_df.columns if c.startswith("feature_")]


def get_selected_model_summary(model_performance_df: pd.DataFrame) -> dict[str, object]:
    """Pick the model performance row with the highest test PR-AUC.

    Args:
        model_performance_df: Output of :func:`load_model_performance`.

    Returns:
        Dictionary for the top row by ``PR_AUC``, or an empty dict if the
        table has no ``PR_AUC`` column / no rows.
    """
    if model_performance_df.empty or "PR_AUC" not in model_performance_df.columns:
        return {}
    best_row = model_performance_df.sort_values("PR_AUC", ascending=False).iloc[0]
    return best_row.to_dict()
