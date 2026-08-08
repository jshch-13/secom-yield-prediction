"""EDA statistics helpers and leakage-safe preprocessing / modeling pipelines.

Design principle: every transformer that has learnable state (imputation
values, which columns to drop, scaling parameters, variance threshold,
SMOTE) is expressed as an ``sklearn``/``imblearn`` pipeline step so that
``.fit()`` is only ever called on the training split, while validation/test
data only ever goes through ``.transform()``. This module exposes both the
plain descriptive-statistics helpers used in the EDA notebook (which may be
computed on the full dataset for exploration purposes only, never for
feature selection feeding into a model) and the model-ready pipeline
factories used by ``src/train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_XGBOOST = False

from src import config

# ---------------------------------------------------------------------------
# Descriptive EDA statistics (exploration only -- not fit/transform state)
# ---------------------------------------------------------------------------


def compute_missing_ratio(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Compute the fraction of missing values per feature column.

    Args:
        df: Source DataFrame.
        feature_cols: Columns to evaluate.

    Returns:
        Series indexed by feature name, sorted descending by missing ratio.
    """
    return df[feature_cols].isna().mean().sort_values(ascending=False)


def compute_variance(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Compute per-feature variance, ignoring missing values.

    Args:
        df: Source DataFrame.
        feature_cols: Columns to evaluate.

    Returns:
        Series indexed by feature name, sorted ascending by variance.
    """
    return df[feature_cols].var(skipna=True).sort_values(ascending=True)


def detect_constant_or_low_variance_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = config.LOW_VARIANCE_THRESHOLD,
) -> list[str]:
    """Identify constant or near-zero-variance feature columns.

    Args:
        df: Source DataFrame.
        feature_cols: Columns to evaluate.
        threshold: Variance below this value is considered "constant" /
            low-variance.

    Returns:
        List of feature names whose variance is below ``threshold`` (NaN
        variance, e.g. an all-missing column, also counts as low-variance).
    """
    variances = compute_variance(df, feature_cols)
    return variances[(variances.isna()) | (variances < threshold)].index.tolist()


def features_above_missing_threshold(
    missing_ratio: pd.Series, threshold: float
) -> list[str]:
    """Return feature names whose missing ratio exceeds ``threshold``.

    Args:
        missing_ratio: Output of :func:`compute_missing_ratio`.
        threshold: Missing-ratio cutoff, e.g. 0.5 for 50%.

    Returns:
        List of feature names to consider dropping at this threshold.
    """
    return missing_ratio[missing_ratio > threshold].index.tolist()


def top_variance_features(
    df: pd.DataFrame, feature_cols: list[str], n: int = config.TOP_N_FEATURES
) -> list[str]:
    """Return the ``n`` highest-variance feature names (for readable heatmaps).

    Args:
        df: Source DataFrame.
        feature_cols: Columns to evaluate.
        n: Number of top-variance features to return.

    Returns:
        List of feature names, highest variance first.
    """
    return compute_variance(df, feature_cols).sort_values(ascending=False).head(n).index.tolist()


def find_high_correlation_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = config.HIGH_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Find feature pairs with absolute Pearson correlation above ``threshold``.

    Args:
        df: Source DataFrame.
        feature_cols: Columns to evaluate.
        threshold: Absolute correlation cutoff.

    Returns:
        DataFrame with columns ``feature_1``, ``feature_2``, ``correlation``,
        sorted by descending absolute correlation.
    """
    corr = df[feature_cols].corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack()
    high_pairs = pairs[pairs.abs() > threshold]
    result = (
        high_pairs.reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
        .sort_values("correlation", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return result


def analyze_time_structure(df: pd.DataFrame, label_col: str = config.LABEL_COL) -> dict[str, Any]:
    """Assess whether a chronological (time-ordered) split looks appropriate.

    This does not decide the split strategy automatically -- it surfaces
    the evidence (whether timestamps are monotonic, and whether the fail
    rate drifts noticeably over time) so that choice can be documented
    explicitly. The project defaults to a stratified random split; a
    time-ordered split is only justified if a clear temporal drift pattern
    is found here.

    Args:
        df: DataFrame with ``Time`` and label columns.
        label_col: Name of the binary label column.

    Returns:
        Dictionary with ``is_monotonic_time``, ``time_span_days``,
        ``fail_rate_by_month`` (as a dict), and ``fail_rate_std_across_months``.
    """
    time_series = df[config.TIME_COL].dropna().sort_values()
    is_monotonic = bool(df[config.TIME_COL].is_monotonic_increasing)
    span_days = (
        float((time_series.max() - time_series.min()).days) if len(time_series) > 1 else 0.0
    )

    monthly = df.dropna(subset=[config.TIME_COL]).copy()
    monthly["_year_month"] = monthly[config.TIME_COL].dt.to_period("M").astype(str)
    fail_rate_by_month = monthly.groupby("_year_month")[label_col].mean().to_dict()
    fail_rate_std = float(np.std(list(fail_rate_by_month.values()))) if fail_rate_by_month else 0.0

    return {
        "is_monotonic_time": is_monotonic,
        "time_span_days": span_days,
        "fail_rate_by_month": fail_rate_by_month,
        "fail_rate_std_across_months": fail_rate_std,
    }


# ---------------------------------------------------------------------------
# Leakage-safe transformers
# ---------------------------------------------------------------------------


class MissingRatioDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missing-value ratio (fit on training data) exceeds a threshold.

    Fitting on the training split only and applying the learned column set
    at transform time prevents information about validation/test missingness
    from influencing which columns are kept.
    """

    def __init__(self, threshold: float = config.MODELING_MISSING_THRESHOLD) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingRatioDropper":
        missing_ratio = X.isna().mean()
        self.columns_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.columns_to_keep_)


class LowVarianceDropper(BaseEstimator, TransformerMixin):
    """NaN-tolerant alternative to sklearn's ``VarianceThreshold``.

    sklearn's built-in ``VarianceThreshold`` rejects input containing NaN,
    which makes it unusable directly ahead of a model with native
    missing-value handling (no imputation step beforehand). This computes
    variance with ``skipna=True`` instead, fit on the training split only.
    """

    def __init__(self, threshold: float = config.LOW_VARIANCE_THRESHOLD) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LowVarianceDropper":
        variances = X.var(skipna=True)
        self.columns_to_keep_ = variances[
            variances.notna() & (variances >= self.threshold)
        ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.columns_to_keep_)


def stratified_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = config.LABEL_COL,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving the Pass/Fail class ratio.

    Args:
        df: Full dataset.
        feature_cols: Sensor feature columns to use as X.
        label_col: Binary label column name.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = df[feature_cols]
    y = df[label_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Model pipeline factory
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Bundle describing one candidate model for comparison."""

    name: str
    preprocessing: str
    sampling: str
    pipeline: ImbPipeline
    param_distributions: dict[str, list[Any]] = field(default_factory=dict)


def _base_steps(missing_threshold: float, use_scaler: bool) -> list[tuple[str, Any]]:
    """Shared leading pipeline steps: missing-ratio drop, imputation, variance filter."""
    steps: list[tuple[str, Any]] = [
        ("missing_dropper", MissingRatioDropper(threshold=missing_threshold)),
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        (
            "variance_threshold",
            VarianceThreshold(threshold=config.LOW_VARIANCE_THRESHOLD).set_output(
                transform="pandas"
            ),
        ),
    ]
    if use_scaler:
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))
    return steps


def build_model_configs(random_state: int = config.RANDOM_STATE) -> list[ModelConfig]:
    """Build the fixed set of candidate preprocessing + model pipelines.

    Four configurations are compared, matching the project's imbalance
    strategy:

    - **A. Baseline_LogisticRegression**: median imputation + constant-feature
      removal + scaling, plain Logistic Regression with no imbalance
      handling at all -- the naive reference point.
    - **B. RandomForest_SMOTE**: same cleaning steps, SMOTE oversampling
      applied only inside the training folds, Random Forest.
    - **C. RandomForest_ClassWeight**: identical Random Forest but using
      ``class_weight="balanced"`` instead of SMOTE, to directly compare
      resampling vs. cost-sensitive weighting on the same model family.
    - **D. XGBoost_ScalePosWeight**: gradient boosting with
      ``scale_pos_weight`` set from the training class ratio, using its
      native handling of missing values (no median imputation) since
      XGBoost splits around missing values natively.

    Args:
        random_state: Seed applied to every estimator and to SMOTE.

    Returns:
        List of :class:`ModelConfig` objects.
    """
    configs: list[ModelConfig] = []

    # A. Baseline: no imbalance handling at all.
    baseline_steps = _base_steps(config.MODELING_MISSING_THRESHOLD, use_scaler=True)
    baseline_steps.append(
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state))
    )
    configs.append(
        ModelConfig(
            name="Baseline_LogisticRegression",
            preprocessing="median_impute+variance_threshold+scale",
            sampling="none",
            pipeline=ImbPipeline(baseline_steps),
            param_distributions={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        )
    )

    # B. Imbalance-aware: SMOTE inside the training folds only.
    smote_steps = _base_steps(config.MODELING_MISSING_THRESHOLD, use_scaler=False)
    smote_steps.append(("smote", SMOTE(random_state=random_state)))
    smote_steps.append(
        ("clf", RandomForestClassifier(random_state=random_state))
    )
    configs.append(
        ModelConfig(
            name="RandomForest_SMOTE",
            preprocessing="median_impute+variance_threshold",
            sampling="SMOTE (train fold only)",
            pipeline=ImbPipeline(smote_steps),
            param_distributions={
                "clf__n_estimators": [200, 300, 400],
                "clf__max_depth": [None, 8, 12],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        )
    )

    # C. Cost-sensitive alternative to SMOTE, same model family for a fair comparison.
    cw_steps = _base_steps(config.MODELING_MISSING_THRESHOLD, use_scaler=False)
    cw_steps.append(
        (
            "clf",
            RandomForestClassifier(
                class_weight="balanced", random_state=random_state
            ),
        )
    )
    configs.append(
        ModelConfig(
            name="RandomForest_ClassWeight",
            preprocessing="median_impute+variance_threshold",
            sampling="class_weight=balanced",
            pipeline=ImbPipeline(cw_steps),
            param_distributions={
                "clf__n_estimators": [200, 300, 400],
                "clf__max_depth": [None, 8, 12],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        )
    )

    # D. Gradient boosting with native missing-value handling + scale_pos_weight.
    if _HAS_XGBOOST:
        gb_steps = [
            ("missing_dropper", MissingRatioDropper(threshold=config.MODELING_MISSING_THRESHOLD)),
            ("variance_threshold", LowVarianceDropper(threshold=config.LOW_VARIANCE_THRESHOLD)),
            (
                "clf",
                XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    missing=np.nan,
                ),
            ),
        ]
        configs.append(
            ModelConfig(
                name="XGBoost_ScalePosWeight",
                preprocessing="variance_threshold_only (native missing handling, no imputation)",
                sampling="scale_pos_weight (from train class ratio)",
                pipeline=ImbPipeline(gb_steps),
                param_distributions={
                    "clf__n_estimators": [200, 300, 400],
                    "clf__max_depth": [3, 4, 5, 6],
                    "clf__learning_rate": [0.01, 0.05, 0.1],
                    "clf__subsample": [0.7, 0.85, 1.0],
                },
            )
        )

    return configs


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute XGBoost's ``scale_pos_weight`` from the training class ratio.

    Args:
        y_train: Training labels (0 = Pass, 1 = Fail).

    Returns:
        Ratio of negative (Pass) to positive (Fail) training samples.
    """
    n_pass = int((y_train == 0).sum())
    n_fail = int((y_train == 1).sum())
    return n_pass / max(n_fail, 1)
