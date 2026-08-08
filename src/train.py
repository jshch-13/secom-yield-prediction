"""End-to-end training entry point.

Run with::

    python src/train.py

Loads raw SECOM data, splits it in a stratified, leakage-safe manner (every
imputer / scaler / variance filter / SMOTE step is fit on the training split
only), tunes four candidate pipelines with a small ``RandomizedSearchCV``
search space, evaluates them on a held-out test set and via stratified
cross-validation, and writes every deliverable this project promises:

- ``outputs/metrics/model_comparison.csv``, ``final_metrics.json``,
  per-model threshold-tuning tables, and ``feature_importance.csv``
- ``outputs/figures/*.png`` (confusion matrices, PR/ROC curves, threshold
  tuning curves, feature importance)
- ``outputs/models/*.joblib`` fitted pipelines
- ``data/processed/secom_spotfire_master.csv``,
  ``secom_feature_summary.csv``, ``secom_model_performance.csv``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, evaluate, preprocess  # noqa: E402
from src.data_loader import get_feature_columns, load_raw_secom  # noqa: E402
from src.utils import ensure_dir, save_json, set_random_seed  # noqa: E402


def run_model_search(
    cfg: preprocess.ModelConfig, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[Any, dict[str, dict[str, float]], dict[str, Any]]:
    """Tune one candidate pipeline with RandomizedSearchCV over StratifiedKFold.

    A small, fixed search space is used deliberately (see
    ``src.preprocess.build_model_configs``) since the training set is small
    and the minority (Fail) class is scarce -- an exhaustive grid search
    would be both slow and prone to overfitting the CV folds. Multiple
    scorers are tracked simultaneously so the CV summary reports the full
    metric set, not just the metric used to pick hyperparameters.

    Args:
        cfg: Candidate pipeline plus its hyperparameter search space.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Tuple of ``(best_fitted_pipeline, cv_summary, best_params)`` where
        ``cv_summary`` maps metric name to ``{"mean": ..., "std": ...}``
        over the CV folds, evaluated at the selected hyperparameters.
    """
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    search = RandomizedSearchCV(
        cfg.pipeline,
        param_distributions=cfg.param_distributions,
        n_iter=config.N_RANDOM_SEARCH_ITER,
        scoring=scoring,
        refit="pr_auc",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_idx = search.best_index_
    cv_summary = {
        metric: {
            "mean": float(search.cv_results_[f"mean_test_{metric}"][best_idx]),
            "std": float(search.cv_results_[f"std_test_{metric}"][best_idx]),
        }
        for metric in scoring
    }
    return search.best_estimator_, cv_summary, search.best_params_


def inject_scale_pos_weight(cfg: preprocess.ModelConfig, y_train: pd.Series) -> None:
    """Set XGBoost's ``scale_pos_weight`` from the training class ratio, if applicable.

    Args:
        cfg: Candidate pipeline; mutated in place if it exposes a
            ``clf__scale_pos_weight`` parameter.
        y_train: Training labels used to compute the class ratio.
    """
    params = cfg.pipeline.get_params()
    if "clf__scale_pos_weight" in params:
        weight = preprocess.compute_scale_pos_weight(y_train)
        cfg.pipeline.set_params(clf__scale_pos_weight=weight)


def export_spotfire_files(
    df: pd.DataFrame,
    feature_cols: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_pipeline: Any,
    best_model_name: str,
    importance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    threshold: float = 0.5,
) -> None:
    """Write the three Spotfire-ready CSVs into ``data/processed``.

    Predictions are generated for both the train and test splits so the
    master table covers every row, but a ``dataset_split`` column is
    included explicitly so a Spotfire user can filter down to ``test`` only
    when judging model performance honestly -- predictions on training rows
    are systematically optimistic and are exposed for coverage/visual
    context, not as evidence of generalization.

    Args:
        df: Full labeled dataset (as returned by ``load_raw_secom``).
        feature_cols: All anonymized sensor feature column names.
        X_train: Training feature matrix (raw, pre-pipeline columns).
        X_test: Test feature matrix (raw, pre-pipeline columns).
        y_train: Training labels.
        y_test: Test labels.
        best_pipeline: Final fitted pipeline used to score every row.
        best_model_name: Name of the final selected model.
        importance_df: Output of ``evaluate.compute_feature_importance``.
        comparison_df: Model comparison table (all candidates).
        threshold: Probability threshold used for ``predicted_class``.
    """
    ensure_dir(config.DATA_PROCESSED_DIR)

    proba_train = best_pipeline.predict_proba(X_train)[:, 1]
    proba_test = best_pipeline.predict_proba(X_test)[:, 1]
    top_features = importance_df["feature_name"].head(config.TOP_N_FEATURES).tolist()

    split_frames = []
    for X_split, proba, split_name in [
        (X_train, proba_train, "train"),
        (X_test, proba_test, "test"),
    ]:
        base_cols = ["row_id", config.TIME_COL, config.LABEL_COL, config.LABEL_TEXT_COL]
        part = df.loc[X_split.index, base_cols].reset_index(drop=True).copy()
        part["dataset_split"] = split_name
        part["model_used"] = best_model_name
        part["fail_probability"] = proba
        part["prediction_threshold"] = threshold
        part["predicted_class"] = (proba >= threshold).astype(int)
        part["model_prediction"] = part["predicted_class"].map({0: "Pass", 1: "Fail"})
        feature_block = df.loc[X_split.index, top_features].reset_index(drop=True)
        split_frames.append(pd.concat([part, feature_block], axis=1))

    master = pd.concat(split_frames, axis=0).sort_values("row_id").reset_index(drop=True)
    master.to_csv(config.DATA_PROCESSED_DIR / "secom_spotfire_master.csv", index=False)

    missing_ratio_full = preprocess.compute_missing_ratio(df, feature_cols)
    variance_full = preprocess.compute_variance(df, feature_cols)
    feature_summary = pd.DataFrame({"feature_name": feature_cols})
    feature_summary["missing_ratio"] = feature_summary["feature_name"].map(missing_ratio_full)
    feature_summary["variance"] = feature_summary["feature_name"].map(variance_full)
    feature_summary = feature_summary.merge(
        importance_df[["feature_name", "importance", "rank"]], on="feature_name", how="left"
    )
    feature_summary = feature_summary.sort_values(
        "importance", ascending=False, na_position="last"
    ).reset_index(drop=True)
    feature_summary.to_csv(config.DATA_PROCESSED_DIR / "secom_feature_summary.csv", index=False)

    comparison_df.to_csv(config.DATA_PROCESSED_DIR / "secom_model_performance.csv", index=False)


def main() -> None:
    """Run the full SECOM training pipeline and write every project deliverable."""
    start = time.time()
    set_random_seed(config.RANDOM_STATE)
    for d in (config.DATA_PROCESSED_DIR, config.FIGURES_DIR, config.METRICS_DIR, config.MODELS_DIR):
        ensure_dir(d)

    print("[1/6] Loading raw SECOM data...")
    df = load_raw_secom(config.DATA_RAW_DIR)
    feature_cols = get_feature_columns(df)
    print(f"  rows={len(df)}, raw feature columns={len(feature_cols)}, "
          f"fail rate={df[config.LABEL_COL].mean():.4f}")

    print("[2/6] Stratified train/test split...")
    X_train, X_test, y_train, y_test = preprocess.stratified_split(df, feature_cols)
    print(f"  train={len(X_train)} (fail rate={y_train.mean():.4f}), "
          f"test={len(X_test)} (fail rate={y_test.mean():.4f})")

    print("[3/6] Fitting & tuning candidate models (RandomizedSearchCV, StratifiedKFold)...")
    model_configs = preprocess.build_model_configs()
    fitted_models: dict[str, Any] = {}
    cv_summaries: dict[str, dict[str, dict[str, float]]] = {}
    best_params: dict[str, dict[str, Any]] = {}
    for cfg in model_configs:
        inject_scale_pos_weight(cfg, y_train)
        print(f"  - {cfg.name}")
        best_pipeline, cv_summary, params = run_model_search(cfg, X_train, y_train)
        fitted_models[cfg.name] = best_pipeline
        cv_summaries[cfg.name] = cv_summary
        best_params[cfg.name] = params

    print("[4/6] Evaluating on held-out test set...")
    comparison_rows = []
    test_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    for cfg in model_configs:
        pipeline = fitted_models[cfg.name]
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate.compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        test_predictions[cfg.name] = (y_test.to_numpy(), y_proba)

        threshold_table = evaluate.threshold_tuning_table(y_test.to_numpy(), y_proba)
        threshold_tables[cfg.name] = threshold_table
        threshold_table.to_csv(
            config.METRICS_DIR / f"threshold_tuning_{cfg.name}.csv", index=False
        )
        evaluate.plot_threshold_curve(threshold_table, cfg.name)
        evaluate.plot_confusion_matrix(y_test.to_numpy(), y_pred, cfg.name)

        comparison_rows.append(
            {
                "Model": cfg.name,
                "Preprocessing": cfg.preprocessing,
                "Sampling_Weighting": cfg.sampling,
                "Recall": metrics["recall"],
                "Precision": metrics["precision"],
                "F1": metrics["f1"],
                "PR_AUC": metrics["pr_auc"],
                "ROC_AUC": metrics["roc_auc"],
                "False_Positives": metrics["false_positives"],
                "False_Negatives": metrics["false_negatives"],
                "CV_PR_AUC_mean": cv_summaries[cfg.name]["pr_auc"]["mean"],
                "CV_PR_AUC_std": cv_summaries[cfg.name]["pr_auc"]["std"],
                "CV_Recall_mean": cv_summaries[cfg.name]["recall"]["mean"],
                "CV_Recall_std": cv_summaries[cfg.name]["recall"]["std"],
            }
        )
        print(
            f"  - {cfg.name}: Recall={metrics['recall']:.3f} Precision={metrics['precision']:.3f} "
            f"F1={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f} ROC-AUC={metrics['roc_auc']:.3f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    evaluate.plot_pr_curve(test_predictions)
    evaluate.plot_roc_curve(test_predictions)

    print("[5/6] Selecting final model (ranked by test PR-AUC) and computing feature importance...")
    best_row = comparison_df.sort_values("PR_AUC", ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_pipeline = fitted_models[best_model_name]
    print(f"  final model: {best_model_name} (test PR-AUC={best_row['PR_AUC']:.4f})")

    importance_df, importance_method = evaluate.compute_feature_importance(
        best_pipeline, X_test, y_test, feature_cols
    )
    importance_df.to_csv(config.METRICS_DIR / "feature_importance.csv", index=False)
    evaluate.plot_feature_importance(importance_df, best_model_name, importance_method)

    best_threshold_table = threshold_tables[best_model_name]
    threshold_recommendations = evaluate.recommend_thresholds(best_threshold_table)

    final_y_true, final_y_proba = test_predictions[best_model_name]
    final_metrics_default = evaluate.compute_metrics(
        final_y_true, (final_y_proba >= 0.5).astype(int), final_y_proba
    )

    final_metrics = {
        "best_model": best_model_name,
        "best_model_preprocessing": best_row["Preprocessing"],
        "best_model_sampling": best_row["Sampling_Weighting"],
        "selection_criterion": (
            "Ranked primarily by held-out test PR-AUC, which is robust to class "
            "imbalance and threshold-independent; Recall, Precision, F1, and the "
            "False Positive count were reviewed jointly rather than optimizing "
            "Recall alone. See reports/model_experiment_log.md for the full "
            "comparison and rationale."
        ),
        "test_metrics_threshold_0.5": final_metrics_default,
        "cross_validation_summary": cv_summaries[best_model_name],
        "threshold_recommendations": threshold_recommendations,
        "importance_method": importance_method,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_fail_rate": float(y_train.mean()),
        "test_fail_rate": float(y_test.mean()),
        "all_models_cv_summary": cv_summaries,
        "all_models_best_params": best_params,
    }
    save_json(final_metrics, config.METRICS_DIR / "final_metrics.json")

    print("[6/6] Saving models, pipeline artifacts, and Spotfire exports...")
    for name, pipeline in fitted_models.items():
        joblib.dump(pipeline, config.MODELS_DIR / f"{name}.joblib")
    joblib.dump(best_pipeline, config.MODELS_DIR / "final_model.joblib")

    export_spotfire_files(
        df,
        feature_cols,
        X_train,
        X_test,
        y_train,
        y_test,
        best_pipeline,
        best_model_name,
        importance_df,
        comparison_df,
    )

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. See outputs/ and data/processed/ for all artifacts.")


if __name__ == "__main__":
    main()
