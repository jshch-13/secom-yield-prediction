"""Project-wide configuration: paths, constants, and reproducibility settings.

All paths are resolved relative to the project root so the pipeline works
regardless of the current working directory it is invoked from.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
MODELS_DIR: Path = OUTPUTS_DIR / "models"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Candidate raw file names, checked in order by data_loader.find_raw_files().
RAW_FEATURES_CANDIDATES: tuple[str, ...] = ("secom.data",)
RAW_LABELS_CANDIDATES: tuple[str, ...] = ("secom_labels.data",)

# Single combined CSV, as distributed by the Kaggle mirror
# "paresh2047/uci-semcom" (kagglehub download): one file with a header row
# containing Time, one column per anonymized sensor, and a Pass/Fail label
# column. Checked before falling back to the two-file UCI layout above.
RAW_COMBINED_CSV_CANDIDATES: tuple[str, ...] = (
    "uci-secom.csv",
    "secom.csv",
    "uci-semcom.csv",
)
LABEL_COLUMN_NAME_CANDIDATES: tuple[str, ...] = (
    "pass/fail",
    "pass_fail",
    "passfail",
    "response",
    "target",
    "label",
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
N_FEATURES_EXPECTED: int = 590  # SECOM has 590 anonymized sensor features
FEATURE_PREFIX: str = "feature_"
FEATURE_NAME_DIGITS: int = 3  # feature_000 .. feature_589

TIME_COL: str = "Time"
LABEL_RAW_COL: str = "Pass_Fail_Raw"  # original -1/1 encoding, preserved as-is
LABEL_COL: str = "Pass_Fail"  # binary encoding: 0 = Pass, 1 = Fail
LABEL_TEXT_COL: str = "Pass_Fail_Label"  # human-readable "Pass" / "Fail"

# Original SECOM label convention: -1 -> Pass (normal), 1 -> Fail (defect)
RAW_LABEL_PASS: int = -1
RAW_LABEL_FAIL: int = 1

# ---------------------------------------------------------------------------
# EDA thresholds
# ---------------------------------------------------------------------------
MISSING_RATIO_THRESHOLDS: tuple[float, ...] = (0.4, 0.5, 0.7)
LOW_VARIANCE_THRESHOLD: float = 1e-6  # near-zero variance cutoff
HIGH_CORR_THRESHOLD: float = 0.95

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
N_RANDOM_SEARCH_ITER: int = 20

# Missing-value ratio threshold used to drop features before imputation
# in the modeling pipeline (distinct from the EDA comparison thresholds above).
MODELING_MISSING_THRESHOLD: float = 0.5

# Probability thresholds scanned for the threshold-tuning analysis.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(t, 2) for t in
                                           [i / 100 for i in range(5, 100, 5)])

# Number of top-importance features surfaced in reports / Spotfire exports.
TOP_N_FEATURES: int = 30
