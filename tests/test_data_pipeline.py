"""Tests for data loading, label conversion, and leakage-safe pipeline behavior.

These tests use small synthetic data shaped like SECOM (whitespace-delimited
feature matrix + label/timestamp file, or the single combined-CSV layout)
rather than the real dataset, so they run without requiring the raw SECOM
files to be present. Synthetic here means "structurally similar for testing
purposes" -- it carries no relationship to real semiconductor sensor
behavior and must never be treated as a stand-in for the real EDA/model
results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, preprocess
from src.data_loader import (
    convert_labels,
    find_raw_files,
    generate_feature_names,
    get_feature_columns,
    load_combined_csv,
    load_raw_secom,
)

N_ROWS = 300
N_FEATURES = 30
RNG_SEED = 0


def _make_synthetic_arrays(
    n_rows: int = N_ROWS, n_features: int = N_FEATURES, fail_rate: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic feature matrix and -1/1 label array for tests."""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_rows, n_features))

    # One constant column, to exercise low-variance detection.
    X[:, 0] = 5.0
    # One column highly correlated with another, to exercise corr-pair detection.
    X[:, 2] = X[:, 1] * 2.0 + rng.normal(scale=1e-6, size=n_rows)
    # Inject missingness: one column mostly missing, others sparsely missing.
    X[:, 3] = np.where(rng.random(n_rows) < 0.9, np.nan, X[:, 3])
    missing_mask = rng.random((n_rows, n_features)) < 0.05
    X[missing_mask] = np.nan
    X[:, 0] = 5.0  # keep constant column intact after masking

    n_fail = int(n_rows * fail_rate)
    labels = np.array([1] * n_fail + [-1] * (n_rows - n_fail))
    rng.shuffle(labels)
    return X, labels


def _write_two_file_format(raw_dir: Path, X: np.ndarray, labels: np.ndarray) -> None:
    lines = []
    for row in X:
        lines.append(" ".join("NaN" if np.isnan(v) else f"{v:.6f}" for v in row))
    (raw_dir / "secom.data").write_text("\n".join(lines) + "\n")

    label_lines = []
    for i, lab in enumerate(labels):
        timestamp = f"{(i % 28) + 1:02d}/07/2008 {(i % 24):02d}:00:00"
        label_lines.append(f"{lab}\t{timestamp}")
    (raw_dir / "secom_labels.data").write_text("\n".join(label_lines) + "\n")


def _write_combined_csv(raw_dir: Path, X: np.ndarray, labels: np.ndarray) -> Path:
    n_rows, n_features = X.shape
    df = pd.DataFrame(X, columns=[str(i) for i in range(n_features)])
    df.insert(0, "Time", [f"{(i % 28) + 1:02d}/07/2008 {(i % 24):02d}:00:00" for i in range(n_rows)])
    df["Pass/Fail"] = labels
    path = raw_dir / "uci-secom.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def two_file_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw_two_file"
    raw_dir.mkdir()
    X, labels = _make_synthetic_arrays()
    _write_two_file_format(raw_dir, X, labels)
    return raw_dir


@pytest.fixture()
def combined_csv_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw_combined"
    raw_dir.mkdir()
    X, labels = _make_synthetic_arrays()
    _write_combined_csv(raw_dir, X, labels)
    return raw_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def test_find_raw_files_two_file_format(two_file_raw_dir: Path) -> None:
    paths = find_raw_files(two_file_raw_dir)
    assert paths["features"].name == "secom.data"
    assert paths["labels"].name == "secom_labels.data"


def test_find_raw_files_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_raw_files(tmp_path / "does_not_exist")


def test_load_raw_secom_two_file_shape_and_columns(two_file_raw_dir: Path) -> None:
    df = load_raw_secom(two_file_raw_dir)
    assert len(df) == N_ROWS
    feature_cols = get_feature_columns(df)
    assert len(feature_cols) == N_FEATURES
    assert feature_cols[0] == "feature_000"
    assert feature_cols[-1] == f"feature_{N_FEATURES - 1:03d}"
    for col in ("row_id", config.TIME_COL, config.LABEL_RAW_COL, config.LABEL_COL, config.LABEL_TEXT_COL):
        assert col in df.columns


def test_load_raw_secom_combined_csv_matches_two_file(
    two_file_raw_dir: Path, combined_csv_raw_dir: Path
) -> None:
    df_two_file = load_raw_secom(two_file_raw_dir)
    df_combined = load_raw_secom(combined_csv_raw_dir)
    assert len(df_two_file) == len(df_combined)
    assert list(get_feature_columns(df_two_file)) == list(get_feature_columns(df_combined))
    # Both fixtures were built from the same underlying synthetic arrays / seed.
    np.testing.assert_allclose(
        df_two_file["feature_001"].to_numpy(),
        df_combined["feature_001"].to_numpy(),
        rtol=1e-4,
    )
    assert (df_two_file[config.LABEL_COL] == df_combined[config.LABEL_COL]).all()


def test_load_combined_csv_direct(combined_csv_raw_dir: Path) -> None:
    csv_path = combined_csv_raw_dir / "uci-secom.csv"
    df = load_combined_csv(csv_path)
    assert len(df) == N_ROWS
    assert set(get_feature_columns(df)) == {f"feature_{i:03d}" for i in range(N_FEATURES)}


def test_generate_feature_names_width_adapts() -> None:
    names_small = generate_feature_names(5)
    assert names_small == [f"feature_{i:03d}" for i in range(5)]
    names_large = generate_feature_names(1200)
    assert names_large[-1] == "feature_1199"
    assert len(names_large[-1]) == len("feature_") + 4


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------


def test_convert_labels_maps_pass_fail_correctly() -> None:
    df = pd.DataFrame({config.LABEL_RAW_COL: [-1, 1, -1, -1, 1]})
    out = convert_labels(df)
    assert out[config.LABEL_COL].tolist() == [0, 1, 0, 0, 1]
    assert out[config.LABEL_TEXT_COL].tolist() == ["Pass", "Fail", "Pass", "Pass", "Fail"]


def test_convert_labels_rejects_unexpected_values() -> None:
    df = pd.DataFrame({config.LABEL_RAW_COL: [-1, 1, 0]})
    with pytest.raises(ValueError):
        convert_labels(df)


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing
# ---------------------------------------------------------------------------


def test_missing_ratio_dropper_fits_on_train_only() -> None:
    train = pd.DataFrame(
        {
            "mostly_missing": [np.nan] * 90 + [1.0] * 10,  # 90% missing in train
            "always_present": np.arange(100, dtype=float),
        }
    )
    # In "test", the same column is fully present -- the dropper must not
    # peek at this when deciding what to drop; the decision is frozen at fit time.
    test = pd.DataFrame(
        {
            "mostly_missing": np.arange(20, dtype=float),
            "always_present": np.arange(20, dtype=float),
        }
    )

    dropper = preprocess.MissingRatioDropper(threshold=0.5)
    dropper.fit(train)
    assert dropper.columns_to_keep_ == ["always_present"]

    transformed_test = dropper.transform(test)
    assert list(transformed_test.columns) == ["always_present"]


def test_low_variance_dropper_handles_nan_and_drops_constant() -> None:
    train = pd.DataFrame(
        {
            "constant": [5.0] * 50,
            "varies": np.concatenate([np.arange(40, dtype=float), [np.nan] * 10]),
        }
    )
    dropper = preprocess.LowVarianceDropper(threshold=config.LOW_VARIANCE_THRESHOLD)
    dropper.fit(train)
    assert "constant" not in dropper.columns_to_keep_
    assert "varies" in dropper.columns_to_keep_


def test_stratified_split_preserves_class_ratio(two_file_raw_dir: Path) -> None:
    df = load_raw_secom(two_file_raw_dir)
    feature_cols = get_feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.stratified_split(
        df, feature_cols, test_size=0.25, random_state=config.RANDOM_STATE
    )
    full_rate = df[config.LABEL_COL].mean()
    assert abs(y_train.mean() - full_rate) < 0.05
    assert abs(y_test.mean() - full_rate) < 0.05
    assert len(X_train) + len(X_test) == len(df)


def test_model_pipeline_fits_and_predicts_without_error(two_file_raw_dir: Path) -> None:
    """End-to-end smoke test: a full candidate pipeline can fit on train and score test.

    This exercises imputation, variance filtering, SMOTE, and the classifier
    together on the small synthetic dataset, confirming the pipeline plumbing
    (leakage-safe fit/transform split, SMOTE applied only inside .fit()) runs
    without raising.
    """
    df = load_raw_secom(two_file_raw_dir)
    feature_cols = get_feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.stratified_split(
        df, feature_cols, test_size=0.3, random_state=config.RANDOM_STATE
    )

    configs = preprocess.build_model_configs()
    baseline_cfg = next(c for c in configs if c.name == "Baseline_LogisticRegression")
    smote_cfg = next(c for c in configs if c.name == "RandomForest_SMOTE")

    baseline_cfg.pipeline.fit(X_train, y_train)
    baseline_proba = baseline_cfg.pipeline.predict_proba(X_test)[:, 1]
    assert baseline_proba.shape[0] == len(X_test)
    assert ((baseline_proba >= 0) & (baseline_proba <= 1)).all()

    train_fail_count_before = int(y_train.sum())
    smote_cfg.pipeline.fit(X_train, y_train)
    # SMOTE must not have mutated the caller's y_train (it resamples internally).
    assert int(y_train.sum()) == train_fail_count_before
    smote_proba = smote_cfg.pipeline.predict_proba(X_test)[:, 1]
    assert smote_proba.shape[0] == len(X_test)


def test_compute_scale_pos_weight() -> None:
    y_train = pd.Series([0] * 80 + [1] * 20)
    weight = preprocess.compute_scale_pos_weight(y_train)
    assert weight == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# EDA helper sanity checks
# ---------------------------------------------------------------------------


def test_missing_ratio_and_constant_detection(two_file_raw_dir: Path) -> None:
    df = load_raw_secom(two_file_raw_dir)
    feature_cols = get_feature_columns(df)

    missing_ratio = preprocess.compute_missing_ratio(df, feature_cols)
    assert missing_ratio["feature_003"] > 0.5

    constant_features = preprocess.detect_constant_or_low_variance_features(df, feature_cols)
    assert "feature_000" in constant_features


def test_high_correlation_pairs_detected(two_file_raw_dir: Path) -> None:
    df = load_raw_secom(two_file_raw_dir)
    feature_cols = get_feature_columns(df)
    pairs = preprocess.find_high_correlation_pairs(df, feature_cols, threshold=0.9)
    pair_names = {frozenset((r.feature_1, r.feature_2)) for r in pairs.itertuples()}
    assert frozenset(("feature_001", "feature_002")) in pair_names
