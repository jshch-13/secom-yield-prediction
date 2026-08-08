"""Generate standalone Plotly HTML reports from real SECOM training results.

Every chart in this module is built strictly from ``data/processed/`` and
``outputs/metrics/`` -- artifacts produced by ``src/train.py`` against the
real SECOM dataset.

Run as a script to (re)generate all four files under ``outputs/interactive/``:

    python dashboard/build_static_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.data_loader import (  # noqa: E402
    get_selected_model_summary,
    load_feature_summary,
    load_master_table,
    load_model_performance,
)
from src import config  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_INTERACTIVE_DIR = PROJECT_ROOT / "outputs" / "interactive"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

TOP_N_DISTRIBUTION = 8
TOP_N_SCATTER_MATRIX = 5
TOP_N_IMPORTANCE = 20
SCATTER_MATRIX_MAX_ROWS = 500
MISCLASSIFICATION_TABLE_MAX_ROWS = 150

CAUSALITY_NOTE = "Importance does not establish physical causality; feature IDs are anonymized SECOM sensor variables."


def build_feature_distribution_html(
    master_df: pd.DataFrame, feature_summary_df: pd.DataFrame, output_path: Path
) -> Path:
    """Build a dropdown-driven Pass/Fail distribution plot for top anonymized sensors.

    Args:
        master_df: Output of ``load_master_table()``.
        feature_summary_df: Output of ``load_feature_summary()``.
        output_path: Destination HTML path.

    Returns:
        The ``output_path`` written.
    """
    ranked = feature_summary_df.dropna(subset=["rank"]).sort_values("rank")
    top_features = [f for f in ranked["feature_name"].tolist() if f in master_df.columns][
        :TOP_N_DISTRIBUTION
    ]

    fig = go.Figure()
    for i, feat in enumerate(top_features):
        for cls, color in [("Pass", "#2E9E5B"), ("Fail", "#D64545")]:
            sub = master_df[master_df["Pass_Fail_Label"] == cls]
            fig.add_trace(
                go.Violin(
                    x=[cls] * len(sub),
                    y=sub[feat],
                    name=cls,
                    legendgroup=cls,
                    line_color=color,
                    box_visible=True,
                    points="all",
                    pointpos=0,
                    jitter=0.35,
                    marker=dict(size=4, opacity=0.5),
                    customdata=np.stack([sub["row_id"], sub["fail_probability"]], axis=-1),
                    hovertemplate=(
                        f"feature: {feat}<br>value: %{{y}}<br>class: {cls}"
                        "<br>row_id: %{customdata[0]}<br>fail_probability: %{customdata[1]:.3f}<extra></extra>"
                    ),
                    visible=(i == 0),
                    showlegend=True,
                )
            )

    buttons = []
    for i, feat in enumerate(top_features):
        visible = [False] * (len(top_features) * 2)
        visible[i * 2] = True
        visible[i * 2 + 1] = True
        buttons.append(
            dict(
                label=feat,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Anonymous SECOM Sensor Analysis - {feat} distribution by Pass/Fail"},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.28, yanchor="top")],
        title=dict(text=f"Anonymous SECOM Sensor Analysis - {top_features[0]} distribution by Pass/Fail", y=0.97, yanchor="top"),
        yaxis_title="Sensor value (anonymized; no physical unit implied)",
        xaxis_title="Pass/Fail (actual label)",
        template="plotly_white",
        height=600,
        margin=dict(t=140),
    )
    fig.add_annotation(
        text="Feature IDs are anonymized SECOM sensor variables; no physical process mapping is implied.",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.14,
        showarrow=False,
        font=dict(size=10, color="#666666"),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def build_scatter_matrix_html(
    master_df: pd.DataFrame, feature_summary_df: pd.DataFrame, output_path: Path
) -> Path:
    """Build a scatter matrix (SPLOM) of top sensors, colored by actual class.

    Args:
        master_df: Output of ``load_master_table()``.
        feature_summary_df: Output of ``load_feature_summary()``.
        output_path: Destination HTML path.

    Returns:
        The ``output_path`` written.
    """
    test_df = master_df[master_df["dataset_split"] == "test"].copy()
    ranked = feature_summary_df.dropna(subset=["rank"]).sort_values("rank")
    top_features = [f for f in ranked["feature_name"].tolist() if f in test_df.columns][
        :TOP_N_SCATTER_MATRIX
    ]

    total_n = len(test_df)
    sampled = total_n > SCATTER_MATRIX_MAX_ROWS
    if sampled:
        frac = SCATTER_MATRIX_MAX_ROWS / total_n
        plot_df = test_df.groupby("Pass_Fail_Label", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=config.RANDOM_STATE)
        )
    else:
        plot_df = test_df

    plot_df = plot_df.copy()
    plot_df["prediction_match"] = np.where(
        plot_df["Pass_Fail_Label"] == plot_df["model_prediction"], "Correct", "Incorrect"
    )
    symbol_map = {"Correct": "circle", "Incorrect": "x"}
    color_map = {"Pass": "#2E9E5B", "Fail": "#D64545"}

    dimensions = [dict(label=f, values=plot_df[f]) for f in top_features]
    fig = go.Figure(
        data=go.Splom(
            dimensions=dimensions,
            text=[
                f"row_id={rid}<br>fail_probability={p:.3f}"
                for rid, p in zip(plot_df["row_id"], plot_df["fail_probability"])
            ],
            marker=dict(
                color=plot_df["Pass_Fail_Label"].map(color_map),
                symbol=plot_df["prediction_match"].map(symbol_map),
                size=6,
                line=dict(width=0.5, color="white"),
                opacity=0.75,
            ),
            diagonal=dict(visible=False),
            showupperhalf=True,
        )
    )

    sample_note = (
        f"Stratified sample: {len(plot_df)} of {total_n} test rows (fail-rate preserved)"
        if sampled
        else f"All {total_n} test rows shown (no sampling needed)"
    )
    fig.update_layout(
        title=f"SECOM Top-{len(top_features)} Sensor Scatter Matrix (Test Split)",
        template="plotly_white",
        height=850,
    )
    fig.add_annotation(
        text=(
            "Color = actual Pass/Fail (green=Pass, red=Fail). Marker shape: circle = correct "
            f"prediction, x = incorrect prediction. {sample_note}."
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False,
        font=dict(size=11, color="#444444"),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def build_prediction_probability_dashboard_html(
    master_df: pd.DataFrame, model_performance_df: pd.DataFrame, output_path: Path
) -> Path:
    """Build the fail-probability / PR-curve / threshold-slider / misclassification dashboard.

    Args:
        master_df: Output of ``load_master_table()``.
        model_performance_df: Output of ``load_model_performance()``.
        output_path: Destination HTML path.

    Returns:
        The ``output_path`` written.
    """
    best = get_selected_model_summary(model_performance_df)
    best_model_name = str(best.get("Model", master_df["model_used"].iloc[0]))
    threshold_csv = METRICS_DIR / f"threshold_tuning_{best_model_name}.csv"
    threshold_df = pd.read_csv(threshold_csv).sort_values("threshold").reset_index(drop=True)

    test_df = master_df[master_df["dataset_split"] == "test"].copy()
    y_true = (test_df["Pass_Fail_Label"] == "Fail").astype(int).to_numpy()
    y_proba = test_df["fail_probability"].to_numpy()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)

    test_df["error_type"] = np.select(
        [
            (test_df["Pass_Fail_Label"] == "Fail") & (test_df["model_prediction"] == "Fail"),
            (test_df["Pass_Fail_Label"] == "Pass") & (test_df["model_prediction"] == "Pass"),
            (test_df["Pass_Fail_Label"] == "Pass") & (test_df["model_prediction"] == "Fail"),
            (test_df["Pass_Fail_Label"] == "Fail") & (test_df["model_prediction"] == "Pass"),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )
    test_df["fail_probability_display"] = test_df["fail_probability"].round(3)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "table", "colspan": 2}, None]],
        subplot_titles=(
            "Fail Probability Distribution (test split)",
            "Precision-Recall Curve (test split)",
            f"Misclassification Table (threshold=0.5, model={best_model_name})",
        ),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
    )

    pass_probs = test_df.loc[test_df["Pass_Fail_Label"] == "Pass", "fail_probability"]
    fail_probs = test_df.loc[test_df["Pass_Fail_Label"] == "Fail", "fail_probability"]
    fig.add_trace(
        go.Histogram(x=pass_probs, name="Pass (actual)", marker_color="#2E9E5B", opacity=0.65, nbinsx=25),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=fail_probs, name="Fail (actual)", marker_color="#D64545", opacity=0.65, nbinsx=25),
        row=1,
        col=1,
    )
    hist_ymax = float(
        max(
            np.histogram(pass_probs, bins=25, range=(0, 1))[0].max() if len(pass_probs) else 1,
            np.histogram(fail_probs, bins=25, range=(0, 1))[0].max() if len(fail_probs) else 1,
        )
    )

    first_row = threshold_df.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[first_row["threshold"], first_row["threshold"]],
            y=[0, hist_ymax],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Selected threshold",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=recall_curve, y=precision_curve, mode="lines", name="PR curve", line=dict(color="#2F6FED")),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[first_row["recall"]],
            y=[first_row["precision"]],
            mode="markers",
            marker=dict(size=14, color="black", symbol="star"),
            name="Selected threshold point",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Fail probability", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0, 1], row=1, col=2)

    table_columns = ["row_id", "Pass_Fail_Label", "model_prediction", "fail_probability_display", "error_type"]
    header_labels = ["row_id", "Actual", "Predicted", "Fail Probability", "Error Type"]
    default_table_df = test_df.sort_values("fail_probability", ascending=False).head(
        MISCLASSIFICATION_TABLE_MAX_ROWS
    )

    fig.add_trace(
        go.Table(
            header=dict(values=header_labels, fill_color="#2F6FED", font=dict(color="white"), align="left"),
            cells=dict(values=[default_table_df[c].tolist() for c in table_columns], align="left"),
        ),
        row=2,
        col=1,
    )

    # Resolve actual trace indices now that every trace has been added.
    trace_names = [t.name for t in fig.data]
    idx_threshold_line = trace_names.index("Selected threshold")
    idx_pr_marker = trace_names.index("Selected threshold point")

    def _kpi_text(row: pd.Series) -> str:
        return (
            f"Threshold={row['threshold']:.2f}  |  Recall={row['recall']:.3f}  |  "
            f"Precision={row['precision']:.3f}  |  F1={row['f1']:.3f}  |  "
            f"FP={int(row['false_positives'])}  |  FN={int(row['false_negatives'])}"
        )

    base_annotations = list(fig.layout.annotations) + [
        dict(
            text=_kpi_text(first_row),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.22,
            showarrow=False,
            font=dict(size=14, color="#1F2430"),
            bgcolor="#F7F8FA",
            bordercolor="#E0E3E8",
            borderwidth=1,
        )
    ]
    fig.update_layout(annotations=base_annotations)

    frames = []
    steps = []
    for _, row in threshold_df.iterrows():
        frame_annotations = list(base_annotations)
        frame_annotations[-1] = {**frame_annotations[-1], "text": _kpi_text(row)}
        frames.append(
            go.Frame(
                name=f"{row['threshold']:.2f}",
                data=[
                    go.Scatter(x=[row["threshold"], row["threshold"]], y=[0, hist_ymax]),
                    go.Scatter(x=[row["recall"]], y=[row["precision"]]),
                ],
                traces=[idx_threshold_line, idx_pr_marker],
                layout=dict(annotations=frame_annotations),
            )
        )
        steps.append(
            dict(
                method="animate",
                label=f"{row['threshold']:.2f}",
                args=[[f"{row['threshold']:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            )
        )
    fig.frames = frames

    filter_options = ["All", "FP", "FN", "TP", "TN"]
    filtered_cache = {}
    for opt in filter_options:
        subset = test_df if opt == "All" else test_df[test_df["error_type"] == opt]
        subset = subset.sort_values("fail_probability", ascending=False).head(MISCLASSIFICATION_TABLE_MAX_ROWS)
        filtered_cache[opt] = [subset[c].tolist() for c in table_columns]

    table_index = trace_names.index(None) if None in trace_names else len(fig.data) - 1
    filter_buttons = [
        dict(
            label=opt,
            method="restyle",
            args=[{"cells.values": [filtered_cache[opt]]}, [table_index]],
        )
        for opt in filter_options
    ]

    fig.update_layout(
        updatemenus=[
            dict(buttons=filter_buttons, direction="down", x=1.0, xanchor="right", y=0.365, yanchor="bottom", showactive=True),
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Threshold: "),
                steps=steps,
                x=0.1,
                len=0.85,
                y=1.42,
            )
        ],
        title=dict(text="SECOM Prediction Probability Dashboard - Threshold trade-off: defect escape vs false alarm", y=0.99, yanchor="top"),
        template="plotly_white",
        height=1100,
        margin=dict(t=260),
        barmode="overlay",
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True, auto_play=False)
    return output_path


def build_feature_importance_html(importance_df: pd.DataFrame, output_path: Path) -> Path:
    """Build a horizontal bar chart of the top-N feature importances.

    Args:
        importance_df: ``outputs/metrics/feature_importance.csv`` contents.
        output_path: Destination HTML path.

    Returns:
        The ``output_path`` written.
    """
    top = importance_df.sort_values("rank").head(TOP_N_IMPORTANCE).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature_name"],
            orientation="h",
            marker_color="#2F6FED",
            hovertemplate="feature: %{y}<br>importance: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top {TOP_N_IMPORTANCE} Feature Importance (Statistical Ranking, Anonymized Sensor IDs)",
        xaxis_title="Importance",
        yaxis_title="Feature (anonymized ID)",
        template="plotly_white",
        height=650,
    )
    fig.add_annotation(
        text=CAUSALITY_NOTE,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        showarrow=False,
        font=dict(size=11, color="#B00020"),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def main() -> None:
    """Generate all four SECOM-based interactive HTML reports."""
    OUTPUTS_INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    master_df = load_master_table()
    feature_summary_df = load_feature_summary()
    model_performance_df = load_model_performance()
    importance_df = pd.read_csv(METRICS_DIR / "feature_importance.csv")

    paths = [
        build_feature_distribution_html(
            master_df, feature_summary_df, OUTPUTS_INTERACTIVE_DIR / "interactive_feature_distribution.html"
        ),
        build_scatter_matrix_html(
            master_df, feature_summary_df, OUTPUTS_INTERACTIVE_DIR / "interactive_scatter_matrix.html"
        ),
        build_prediction_probability_dashboard_html(
            master_df, model_performance_df, OUTPUTS_INTERACTIVE_DIR / "prediction_probability_dashboard.html"
        ),
        build_feature_importance_html(importance_df, OUTPUTS_INTERACTIVE_DIR / "feature_importance.html"),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
