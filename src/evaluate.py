"""Metrics, threshold tuning, feature-importance, and plotting utilities.

All figures are saved as PNG files under ``outputs/figures`` via
``src.utils.save_current_figure``. Metric dictionaries are JSON-serializable
(NumPy scalars are converted) so they can be written directly with
``src.utils.save_json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config
from src.utils import save_current_figure

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Compute the defect-detection-focused metric set at a fixed threshold.

    Accuracy is intentionally not treated as the headline metric given the
    severe class imbalance in SECOM (far more Pass than Fail); Recall,
    Precision, F1, PR-AUC, and ROC-AUC together with the raw confusion
    counts are reported instead.

    Args:
        y_true: Ground-truth binary labels (0 = Pass, 1 = Fail).
        y_pred: Predicted binary labels at the evaluated threshold.
        y_proba: Predicted probability of the Fail class, used for the
            threshold-independent PR-AUC / ROC-AUC.

    Returns:
        Dictionary with ``recall``, ``precision``, ``f1``, ``pr_auc``,
        ``roc_auc``, ``accuracy``, ``confusion_matrix`` (nested list, rows =
        true class), ``true_negatives``, ``false_positives``,
        ``false_negatives``, ``true_positives``.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def threshold_tuning_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: tuple[float, ...] = config.THRESHOLD_GRID,
) -> pd.DataFrame:
    """Sweep classification thresholds and report Recall/Precision/F1 at each.

    Recall and Precision trade off directly as the threshold moves: a lower
    threshold flags more Fail predictions (higher Recall, more false
    positives / inspection cost); a higher threshold is more conservative
    (higher Precision, more missed defects). This table makes that
    trade-off explicit instead of only reporting the default 0.5 cutoff.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probability of the Fail class.
        thresholds: Threshold values to evaluate.

    Returns:
        DataFrame with one row per threshold: ``threshold``, ``recall``,
        ``precision``, ``f1``, ``false_positives``, ``false_negatives``,
        ``true_positives``, ``true_negatives``.
    """
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "threshold": t,
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )
    return pd.DataFrame(rows)


def recommend_thresholds(threshold_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Suggest thresholds for a few common business objectives.

    This is decision support, not an automatic choice: the final threshold
    still has to be picked with the actual cost of a missed defect (false
    negative) vs. an unnecessary inspection (false positive) in mind.

    Args:
        threshold_table: Output of :func:`threshold_tuning_table`.

    Returns:
        Dictionary keyed by strategy name (``"max_f1"``,
        ``"high_recall_min_precision_0.2"``, ``"high_precision_min_recall_0.5"``),
        each mapping to the corresponding row of ``threshold_table`` as a dict.
    """
    recommendations: dict[str, dict[str, Any]] = {}

    best_f1_row = threshold_table.loc[threshold_table["f1"].idxmax()]
    recommendations["max_f1"] = best_f1_row.to_dict()

    high_recall = threshold_table[threshold_table["precision"] >= 0.2]
    if not high_recall.empty:
        row = high_recall.loc[high_recall["recall"].idxmax()]
        recommendations["high_recall_min_precision_0.2"] = row.to_dict()

    high_precision = threshold_table[threshold_table["recall"] >= 0.5]
    if not high_precision.empty:
        row = high_precision.loc[high_precision["precision"].idxmax()]
        recommendations["high_precision_min_recall_0.5"] = row.to_dict()

    return recommendations


# ---------------------------------------------------------------------------
# Cross-validation summary
# ---------------------------------------------------------------------------


def summarize_cv_scores(cv_results: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Summarize StratifiedKFold cross-validation scores as mean +/- std.

    Args:
        cv_results: Mapping of metric name to an array of per-fold scores
            (e.g. as produced by ``sklearn.model_selection.cross_validate``).

    Returns:
        Dictionary keyed by metric name, each mapping to
        ``{"mean": ..., "std": ...}``.
    """
    return {
        metric: {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        for metric, scores in cv_results.items()
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def compute_feature_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    random_state: int = config.RANDOM_STATE,
    n_repeats: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Compute feature importance, preferring SHAP and falling back to permutation importance.

    Importance is reported purely as a statistical ranking of anonymized
    ``feature_XXX`` IDs. It must not be interpreted as identifying a
    physical equipment, chamber, or process step, since SECOM's feature
    identities are anonymized and no such mapping exists in this dataset.

    Args:
        pipeline: Fitted end-to-end pipeline (preprocessing + classifier).
        X_test: Held-out feature matrix (raw, pre-pipeline columns).
        y_test: Held-out labels.
        feature_names: Names of the columns in ``X_test``, in order.
        random_state: Seed for permutation importance shuffling.
        n_repeats: Number of shuffles per feature for permutation importance.

    Returns:
        Tuple of (importance DataFrame with columns ``feature_name``,
        ``importance``, ``rank``; and the method name used, either
        ``"shap"`` or ``"permutation_importance"``).
    """
    try:
        import shap

        clf = pipeline.named_steps.get("clf", pipeline.steps[-1][1])
        pre = pipeline[:-1]
        X_transformed = pre.transform(X_test)
        sample = X_transformed.sample(
            n=min(200, len(X_transformed)), random_state=random_state
        )
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        cols = (
            X_transformed.columns
            if hasattr(X_transformed, "columns")
            else feature_names[: mean_abs_shap.shape[0]]
        )
        importance_df = pd.DataFrame({"feature_name": cols, "importance": mean_abs_shap})
        method = "shap"
    except Exception:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="average_precision",
            n_jobs=-1,
        )
        importance_df = pd.DataFrame(
            {"feature_name": feature_names, "importance": result.importances_mean}
        )
        method = "permutation_importance"

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    return importance_df, method


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot and save a confusion matrix for one model.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        model_name: Used in the title and output filename.

    Returns:
        Path to the saved PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Pass", "Fail"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    path = config.FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    save_current_figure(path)
    return path


def plot_pr_curve(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Plot Precision-Recall curves for one or more models on one axes.

    Args:
        results: Mapping of model name to ``(y_true, y_proba)``.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best", fontsize=8)
    path = config.FIGURES_DIR / "precision_recall_curve.png"
    save_current_figure(path)
    return path


def plot_roc_curve(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Plot ROC curves for one or more models on one axes.

    Args:
        results: Mapping of model name to ``(y_true, y_proba)``.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="best", fontsize=8)
    path = config.FIGURES_DIR / "roc_curve.png"
    save_current_figure(path)
    return path


def plot_threshold_curve(threshold_table: pd.DataFrame, model_name: str) -> Path:
    """Plot Recall/Precision/F1 as a function of the classification threshold.

    Args:
        threshold_table: Output of :func:`threshold_tuning_table`.
        model_name: Used in the title and output filename.

    Returns:
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric in ["recall", "precision", "f1"]:
        ax.plot(threshold_table["threshold"], threshold_table[metric], marker="o", markersize=3, label=metric)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning - {model_name}")
    ax.legend(loc="best")
    path = config.FIGURES_DIR / f"threshold_tuning_{model_name}.png"
    save_current_figure(path)
    return path


def plot_feature_importance(
    importance_df: pd.DataFrame, model_name: str, method: str, top_n: int = 20
) -> Path:
    """Plot a horizontal bar chart of the top-N most important features.

    Args:
        importance_df: Output of :func:`compute_feature_importance`.
        model_name: Used in the title and output filename.
        method: Importance method name (``"shap"`` or ``"permutation_importance"``),
            shown in the subtitle as a reminder that this is a statistical
            ranking, not a physical/process interpretation.
        top_n: Number of top features to display.

    Returns:
        Path to the saved PNG.
    """
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    ax.barh(top["feature_name"], top["importance"], color="#4C72B0")
    ax.set_xlabel("Importance (statistical ranking, anonymized feature ID)")
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}\n(method: {method})")
    path = config.FIGURES_DIR / f"feature_importance_{model_name}.png"
    save_current_figure(path)
    return path
