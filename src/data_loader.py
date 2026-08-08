"""Load raw SECOM sensor data and labels into a single tidy DataFrame.

The canonical UCI SECOM distribution ships two whitespace-delimited files:

- ``secom.data``: 1567 rows x 590 columns of anonymized sensor readings,
  missing values encoded as the literal string ``NaN``.
- ``secom_labels.data``: 1567 rows of ``<label> <date> <time>``, where the
  label is ``-1`` (Pass) or ``1`` (Fail).

The Kaggle mirror ``paresh2047/uci-semcom`` (downloaded via ``kagglehub``)
instead ships a single combined CSV, ``uci-secom.csv``, with a header row:
``Time``, one column per anonymized sensor, and a ``Pass/Fail`` label
column. Both layouts are auto-detected inside ``data/raw`` -- the combined
CSV is tried first, falling back to the two-file layout, and finally to a
pattern-based search if the exact names differ -- and both are normalized
into one merged DataFrame with explicit ``feature_000 .. feature_589``
columns plus ``Time`` and ``Pass_Fail`` columns. No physical meaning is
assumed for any feature column -- SECOM feature IDs are anonymized and are
treated purely as statistical variables throughout this project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

# Observed Time formats across SECOM distributions: the classic UCI
# secom_labels.data uses DD/MM/YYYY, while the Kaggle uci-secom.csv mirror
# uses ISO-like YYYY-MM-DD. Both are tried before falling back to pandas'
# general inference.
_TIME_FORMATS: tuple[str, ...] = ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse a Time column trying known SECOM formats before generic inference.

    Args:
        series: Raw timestamp strings.

    Returns:
        Parsed ``datetime64`` series. Values that cannot be parsed become
        ``NaT`` rather than raising, since ``Time`` is metadata and must
        never block loading the feature/label data.
    """
    for fmt in _TIME_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed
    return pd.to_datetime(series, errors="coerce")


def find_raw_files(raw_dir: Path = config.DATA_RAW_DIR) -> dict[str, Path]:
    """Locate the SECOM feature and label files inside ``raw_dir``.

    Tries the canonical UCI file names first (``secom.data`` /
    ``secom_labels.data``); if either is missing, falls back to scanning
    ``raw_dir`` for ``.data`` / ``.csv`` / ``.txt`` files and guessing based
    on filename (a file with "label" in its name is treated as the label
    file).

    Args:
        raw_dir: Directory expected to contain the raw SECOM files.

    Returns:
        Dictionary with keys ``"features"`` and ``"labels"`` mapping to
        resolved file paths.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist, or no plausible
            features/labels file pair could be identified.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. "
            "See data/README.md for how to download and place the SECOM dataset."
        )

    features_path: Path | None = None
    labels_path: Path | None = None

    for name in config.RAW_FEATURES_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            features_path = candidate
            break

    for name in config.RAW_LABELS_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            labels_path = candidate
            break

    if features_path is None or labels_path is None:
        candidates = sorted(
            {*raw_dir.glob("*.data"), *raw_dir.glob("*.csv"), *raw_dir.glob("*.txt")}
        )
        for f in candidates:
            lname = f.name.lower()
            if "label" in lname and labels_path is None:
                labels_path = f
            elif "label" not in lname and features_path is None:
                features_path = f

    if features_path is None or labels_path is None:
        raise FileNotFoundError(
            f"Could not find both a features file and a labels file under {raw_dir}. "
            f"Expected e.g. {config.RAW_FEATURES_CANDIDATES[0]!r} and "
            f"{config.RAW_LABELS_CANDIDATES[0]!r}. "
            "See data/README.md for the expected layout."
        )

    return {"features": features_path, "labels": labels_path}


def generate_feature_names(n_features: int) -> list[str]:
    """Generate explicit, zero-padded sensor feature column names.

    Args:
        n_features: Number of anonymized sensor columns in the raw file.

    Returns:
        List of names like ``["feature_000", "feature_001", ...]``. The
        zero-padding width expands automatically if ``n_features`` exceeds
        what the default width (``config.FEATURE_NAME_DIGITS``) can express,
        so column names remain lexicographically sortable.
    """
    width = max(config.FEATURE_NAME_DIGITS, len(str(max(n_features - 1, 0))))
    return [f"{config.FEATURE_PREFIX}{i:0{width}d}" for i in range(n_features)]


def load_features(features_path: Path) -> pd.DataFrame:
    """Read the whitespace-delimited sensor feature matrix.

    Args:
        features_path: Path to the raw features file (e.g. ``secom.data``).

    Returns:
        DataFrame with explicit ``feature_XXX`` column names and missing
        values as ``NaN``.
    """
    df = pd.read_csv(features_path, sep=r"\s+", header=None, na_values=["NaN", "nan"])
    df.columns = generate_feature_names(df.shape[1])
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Read the label/timestamp file.

    Each line is ``<label><whitespace><date> <time>``, e.g.
    ``-1\t19/07/2008 11:55:00``. The label and the full timestamp string are
    split on the first whitespace run only, since the timestamp itself
    contains an internal space between date and time.

    Args:
        labels_path: Path to the raw labels file (e.g. ``secom_labels.data``).

    Returns:
        DataFrame with columns ``Pass_Fail_Raw`` (original -1/1 int) and
        ``Time`` (parsed as pandas datetime; original string preserved
        losslessly via standard ISO parsing, no derived calendar features
        are added here -- see the EDA notebook for that discussion).
    """
    records: list[tuple[int, str | None]] = []
    for raw_line in labels_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        label = int(parts[0])
        timestamp = parts[1].strip() if len(parts) > 1 else None
        records.append((label, timestamp))

    labels_df = pd.DataFrame(records, columns=[config.LABEL_RAW_COL, config.TIME_COL])
    labels_df[config.TIME_COL] = _parse_timestamps(labels_df[config.TIME_COL])
    return labels_df


def convert_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw -1/1 SECOM label to a standard binary target.

    SECOM convention: ``-1`` = Pass (normal), ``1`` = Fail (defect). This
    function adds ``Pass_Fail`` (0 = Pass, 1 = Fail) and ``Pass_Fail_Label``
    (human-readable "Pass"/"Fail") columns based on ``Pass_Fail_Raw``.

    Args:
        df: DataFrame containing a ``Pass_Fail_Raw`` column with -1/1 values.

    Returns:
        The same DataFrame with ``Pass_Fail`` and ``Pass_Fail_Label`` columns
        added.

    Raises:
        ValueError: If ``Pass_Fail_Raw`` contains values other than -1/1.
    """
    unexpected = set(df[config.LABEL_RAW_COL].unique()) - {
        config.RAW_LABEL_PASS,
        config.RAW_LABEL_FAIL,
    }
    if unexpected:
        raise ValueError(
            f"Unexpected raw label values {unexpected}; expected only "
            f"{config.RAW_LABEL_PASS} (Pass) and {config.RAW_LABEL_FAIL} (Fail)."
        )

    df = df.copy()
    df[config.LABEL_COL] = (df[config.LABEL_RAW_COL] == config.RAW_LABEL_FAIL).astype(int)
    df[config.LABEL_TEXT_COL] = df[config.LABEL_COL].map({0: "Pass", 1: "Fail"})
    return df


def find_combined_csv(raw_dir: Path = config.DATA_RAW_DIR) -> Path | None:
    """Look for a single combined SECOM CSV (Kaggle ``uci-secom.csv`` layout).

    Tries the known candidate filenames first; if none match, scans every
    ``*.csv`` in ``raw_dir`` and accepts the first one whose header contains
    a recognizable Pass/Fail label column name.

    Args:
        raw_dir: Directory expected to contain the raw SECOM files.

    Returns:
        Path to the combined CSV if found, else ``None``.
    """
    if not raw_dir.exists():
        return None

    for name in config.RAW_COMBINED_CSV_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate

    label_names = set(config.LABEL_COLUMN_NAME_CANDIDATES)
    for csv_path in sorted(raw_dir.glob("*.csv")):
        try:
            header = pd.read_csv(csv_path, nrows=0).columns
        except (pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
        if {c.strip().lower() for c in header} & label_names:
            return csv_path
    return None


def load_combined_csv(csv_path: Path) -> pd.DataFrame:
    """Load the single-file Kaggle ``uci-secom.csv`` layout.

    Expects a header row containing a ``Time`` column, a Pass/Fail label
    column (matched case-insensitively against
    ``config.LABEL_COLUMN_NAME_CANDIDATES``), and one column per anonymized
    sensor. Any leading unnamed index column (as pandas writes when a CSV is
    exported with ``index=True``) is dropped.

    Args:
        csv_path: Path to the combined CSV file.

    Returns:
        DataFrame in the same normalized layout as :func:`load_raw_secom`.

    Raises:
        ValueError: If no recognizable Pass/Fail label column is found.
    """
    raw = pd.read_csv(csv_path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.match(r"^Unnamed")]

    time_col = next((c for c in raw.columns if str(c).strip().lower() == "time"), None)
    label_names = set(config.LABEL_COLUMN_NAME_CANDIDATES)
    label_col = next(
        (c for c in raw.columns if str(c).strip().lower() in label_names), None
    )
    if label_col is None:
        raise ValueError(
            f"Could not find a Pass/Fail label column in {csv_path}. "
            f"Expected one of {config.LABEL_COLUMN_NAME_CANDIDATES!r} in the header."
        )

    feature_source_cols = [c for c in raw.columns if c not in {time_col, label_col}]
    features_df = raw[feature_source_cols].apply(pd.to_numeric, errors="coerce")
    features_df.columns = generate_feature_names(features_df.shape[1])

    out = pd.DataFrame(index=raw.index)
    out[config.TIME_COL] = (
        _parse_timestamps(raw[time_col]) if time_col is not None else pd.NaT
    )
    out[config.LABEL_RAW_COL] = raw[label_col].astype(int)
    out = pd.concat([out.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    out = convert_labels(out)
    out.insert(0, "row_id", range(len(out)))

    ordered_cols = [
        "row_id",
        config.TIME_COL,
        config.LABEL_RAW_COL,
        config.LABEL_COL,
        config.LABEL_TEXT_COL,
    ] + list(features_df.columns)
    return out[ordered_cols]


def load_raw_secom(raw_dir: Path = config.DATA_RAW_DIR) -> pd.DataFrame:
    """Load, merge, and label-encode the full raw SECOM dataset.

    Auto-detects whether ``raw_dir`` holds the single combined CSV layout
    (Kaggle ``uci-secom.csv``) or the two-file UCI layout (``secom.data`` +
    ``secom_labels.data``) and dispatches accordingly.

    Args:
        raw_dir: Directory containing the raw SECOM files.

    Returns:
        DataFrame indexed by ``row_id`` with columns, in order: ``row_id``,
        ``Time``, ``Pass_Fail_Raw``, ``Pass_Fail``, ``Pass_Fail_Label``,
        followed by ``feature_000 .. feature_{n-1}``.

    Raises:
        FileNotFoundError: If raw files cannot be located under ``raw_dir``.
        ValueError: If the features and labels files have mismatched row
            counts, or labels contain unexpected values.
    """
    combined_path = find_combined_csv(raw_dir)
    if combined_path is not None:
        return load_combined_csv(combined_path)

    paths = find_raw_files(raw_dir)
    features_df = load_features(paths["features"])
    labels_df = load_labels(paths["labels"])

    if len(features_df) != len(labels_df):
        raise ValueError(
            f"Row count mismatch between features ({len(features_df)}) and "
            f"labels ({len(labels_df)}); files may be corrupted or mismatched."
        )

    merged = pd.concat(
        [labels_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )
    merged = convert_labels(merged)
    merged.insert(0, "row_id", range(len(merged)))

    ordered_cols = [
        "row_id",
        config.TIME_COL,
        config.LABEL_RAW_COL,
        config.LABEL_COL,
        config.LABEL_TEXT_COL,
    ] + list(features_df.columns)
    return merged[ordered_cols]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of columns that are anonymized sensor features.

    Args:
        df: DataFrame produced by :func:`load_raw_secom`.

    Returns:
        List of column names starting with ``config.FEATURE_PREFIX``.
    """
    return [c for c in df.columns if c.startswith(config.FEATURE_PREFIX)]
