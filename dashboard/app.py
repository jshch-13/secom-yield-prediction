"""SECOM Yield Prediction Analytics Dashboard (Plotly Dash).

A read-only viewer over ``data/processed/secom_spotfire_master.csv`` and
``secom_model_performance.csv`` -- the artifacts ``src/train.py`` produces
from the real SECOM dataset. This app never recomputes predictions itself;
it only filters and visualizes what training already produced, so the
numbers shown here always match ``outputs/metrics/`` and the README.

Run with::

    python dashboard/app.py

Then open http://127.0.0.1:8050 in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.components import (  # noqa: E402
    COLORS,
    build_chart_card,
    build_disclaimer_banner,
    build_filter_panel,
    build_kpi_card,
    build_kpi_row,
)
from dashboard.data_loader import (  # noqa: E402
    DashboardDataError,
    data_is_available,
    get_feature_columns,
    get_selected_model_summary,
    load_feature_summary,
    load_master_table,
    load_model_performance,
)

APP_TITLE = "SECOM Yield Prediction Analytics Dashboard"


def _build_no_data_layout() -> html.Div:
    """Layout shown when the required processed CSVs are missing.

    Returns:
        A styled ``html.Div`` explaining how to generate the missing data.
    """
    return html.Div(
        [
            html.H2(APP_TITLE),
            html.Div(
                "No processed data found.",
                style={"color": COLORS["fail_color"], "fontWeight": 700, "fontSize": "16px", "marginBottom": "8px"},
            ),
            html.P(
                "Run `python src/train.py` (with real SECOM data placed under data/raw/, "
                "see data/README.md) to generate data/processed/secom_spotfire_master.csv, "
                "secom_feature_summary.csv, and secom_model_performance.csv, then restart "
                "this dashboard."
            ),
        ],
        style={"padding": "40px", "fontFamily": "Arial, sans-serif"},
    )


def _create_app() -> Dash:
    """Build and configure the Dash application.

    Returns:
        The configured ``Dash`` app instance (not yet running).
    """
    app = Dash(__name__, title=APP_TITLE)

    if not data_is_available():
        app.layout = _build_no_data_layout()
        return app

    try:
        master_df = load_master_table()
        feature_summary_df = load_feature_summary()
        model_performance_df = load_model_performance()
    except DashboardDataError as exc:
        app.layout = html.Div(
            [html.H2(APP_TITLE), html.P(str(exc), style={"color": COLORS["fail_color"]})],
            style={"padding": "40px", "fontFamily": "Arial, sans-serif"},
        )
        return app

    feature_columns = get_feature_columns(master_df)
    ranked_features = (
        feature_summary_df.dropna(subset=["rank"]).sort_values("rank")["feature_name"].tolist()
    )
    ranked_features = [f for f in ranked_features if f in feature_columns] or feature_columns
    default_features = ranked_features[:2] if len(ranked_features) >= 2 else ranked_features

    best_model = get_selected_model_summary(model_performance_df)
    best_model_name = str(best_model.get("Model", master_df["model_used"].iloc[0]))
    selected_threshold = float(master_df["prediction_threshold"].iloc[0])

    master_df["error_type"] = np.select(
        [
            (master_df["Pass_Fail_Label"] == "Fail") & (master_df["model_prediction"] == "Fail"),
            (master_df["Pass_Fail_Label"] == "Pass") & (master_df["model_prediction"] == "Pass"),
            (master_df["Pass_Fail_Label"] == "Pass") & (master_df["model_prediction"] == "Fail"),
            (master_df["Pass_Fail_Label"] == "Fail") & (master_df["model_prediction"] == "Pass"),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )

    total_samples = len(master_df)
    fail_count = int((master_df["Pass_Fail_Label"] == "Fail").sum())
    fail_rate = fail_count / total_samples if total_samples else 0.0

    kpi_row = build_kpi_row(
        [
            build_kpi_card("Total Samples", f"{total_samples:,}"),
            build_kpi_card("Fail Count", f"{fail_count:,}"),
            build_kpi_card("Fail Rate", f"{fail_rate:.2%}"),
            build_kpi_card("Selected Model Recall", f"{float(best_model.get('Recall', 0)):.3f}"),
            build_kpi_card("Precision", f"{float(best_model.get('Precision', 0)):.3f}"),
            build_kpi_card("PR-AUC", f"{float(best_model.get('PR_AUC', 0)):.3f}"),
            build_kpi_card("Selected Threshold", f"{selected_threshold:.2f}"),
        ]
    )

    app.layout = html.Div(
        [
            html.H2(APP_TITLE, style={"marginBottom": "4px"}),
            html.Div(
                f"Model shown: {best_model_name} | Data source: data/processed/secom_spotfire_master.csv",
                style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginBottom": "12px"},
            ),
            build_disclaimer_banner(),
            kpi_row,
            build_filter_panel(feature_columns, default_features),
            html.Div(
                [
                    build_chart_card("Sensor Distribution by Pass/Fail", dcc.Graph(id="chart-sensor-box")),
                    build_chart_card("Fail Probability Distribution", dcc.Graph(id="chart-fail-prob-hist")),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    build_chart_card("Actual vs Predicted Confusion Matrix", dcc.Graph(id="chart-confusion-matrix")),
                    build_chart_card("Feature Importance (Top 20)", dcc.Graph(id="chart-feature-importance")),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    build_chart_card("Selected-Sensor Scatter Plot", dcc.Graph(id="chart-scatter")),
                    build_chart_card("Threshold-Performance Curve", dcc.Graph(id="chart-threshold-curve")),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        "Detail Table",
                        style={"fontSize": "14px", "fontWeight": 600, "marginBottom": "8px"},
                    ),
                    html.Button("Download CSV", id="btn-download-csv", style={"marginBottom": "8px"}),
                    dcc.Download(id="download-detail-csv"),
                    dash_table.DataTable(
                        id="detail-table",
                        columns=[
                            {"name": c, "id": c}
                            for c in [
                                "row_id",
                                "Pass_Fail_Label",
                                "model_prediction",
                                "fail_probability",
                                "error_type",
                                *default_features,
                            ]
                        ],
                        page_size=12,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": "Arial, sans-serif", "fontSize": "12px", "padding": "6px"},
                        style_header={"backgroundColor": COLORS["accent"], "color": "white", "fontWeight": 600},
                        sort_action="native",
                    ),
                ],
                style={
                    "backgroundColor": COLORS["card_background"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "10px",
                    "padding": "16px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["background"],
            "padding": "20px 28px",
            "fontFamily": "Arial, sans-serif",
            "color": COLORS["text_primary"],
            "minHeight": "100vh",
        },
    )

    def _filtered_df(actual_class: str, predicted_class: str, error_type: str, prob_range: list[float]) -> pd.DataFrame:
        df = master_df
        if actual_class != "ALL":
            df = df[df["Pass_Fail_Label"] == actual_class]
        if predicted_class != "ALL":
            df = df[df["model_prediction"] == predicted_class]
        if error_type != "All":
            df = df[df["error_type"] == error_type]
        lo, hi = prob_range
        df = df[(df["fail_probability"] >= lo) & (df["fail_probability"] <= hi)]
        return df

    @app.callback(
        Output("chart-sensor-box", "figure"),
        Output("chart-fail-prob-hist", "figure"),
        Output("chart-confusion-matrix", "figure"),
        Output("chart-feature-importance", "figure"),
        Output("chart-scatter", "figure"),
        Output("chart-threshold-curve", "figure"),
        Output("detail-table", "data"),
        Output("detail-table", "columns"),
        Input("filter-actual-class", "value"),
        Input("filter-predicted-class", "value"),
        Input("filter-error-type", "value"),
        Input("filter-fail-probability", "value"),
        Input("filter-selected-features", "value"),
    )
    def _update_all(actual_class, predicted_class, error_type, prob_range, selected_features):
        selected_features = selected_features or default_features
        filtered = _filtered_df(actual_class, predicted_class, error_type, prob_range)

        if filtered.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No rows match the current filters.", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper"
            )
            empty_fig.update_layout(template="plotly_white")
            empty_cols = [{"name": c, "id": c} for c in ["row_id", "Pass_Fail_Label", "model_prediction", "fail_probability", "error_type"]]
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, [], empty_cols

        box_feature = selected_features[0] if selected_features else feature_columns[0]
        box_fig = px.box(
            filtered,
            x="Pass_Fail_Label",
            y=box_feature,
            color="Pass_Fail_Label",
            color_discrete_map={"Pass": COLORS["pass_color"], "Fail": COLORS["fail_color"]},
            points="outliers",
            template="plotly_white",
        )
        box_fig.update_layout(showlegend=False, xaxis_title="Actual Pass/Fail", yaxis_title=box_feature)

        hist_fig = px.histogram(
            filtered,
            x="fail_probability",
            color="Pass_Fail_Label",
            color_discrete_map={"Pass": COLORS["pass_color"], "Fail": COLORS["fail_color"]},
            barmode="overlay",
            opacity=0.65,
            template="plotly_white",
        )
        hist_fig.update_layout(xaxis_title="Fail probability", yaxis_title="Count")

        cm = confusion_matrix(
            filtered["Pass_Fail_Label"], filtered["model_prediction"], labels=["Pass", "Fail"]
        )
        cm_fig = px.imshow(
            cm,
            x=["Pred: Pass", "Pred: Fail"],
            y=["Actual: Pass", "Actual: Fail"],
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        cm_fig.update_layout(coloraxis_showscale=False)

        top_importance = feature_summary_df.dropna(subset=["rank"]).sort_values("rank").head(20).iloc[::-1]
        importance_fig = px.bar(
            top_importance,
            x="importance",
            y="feature_name",
            orientation="h",
            template="plotly_white",
        )
        importance_fig.update_traces(marker_color=COLORS["accent"])
        importance_fig.update_layout(yaxis_title="Feature (anonymized ID)", xaxis_title="Importance")

        if len(selected_features) >= 2:
            x_feat, y_feat = selected_features[0], selected_features[1]
        elif len(selected_features) == 1:
            x_feat = y_feat = selected_features[0]
        else:
            x_feat, y_feat = feature_columns[0], feature_columns[min(1, len(feature_columns) - 1)]
        scatter_fig = px.scatter(
            filtered,
            x=x_feat,
            y=y_feat,
            color="Pass_Fail_Label",
            color_discrete_map={"Pass": COLORS["pass_color"], "Fail": COLORS["fail_color"]},
            hover_data=["row_id", "fail_probability", "model_prediction"],
            template="plotly_white",
        )

        threshold_csv_path = (
            Path(__file__).resolve().parent.parent / "outputs" / "metrics" / f"threshold_tuning_{best_model_name}.csv"
        )
        if threshold_csv_path.exists():
            threshold_df = pd.read_csv(threshold_csv_path)
            threshold_fig = go.Figure()
            for metric, color in [("recall", "#2F6FED"), ("precision", "#D64545"), ("f1", "#2E9E5B")]:
                threshold_fig.add_trace(
                    go.Scatter(x=threshold_df["threshold"], y=threshold_df[metric], mode="lines+markers", name=metric)
                )
            threshold_fig.update_layout(
                template="plotly_white", xaxis_title="Decision Threshold", yaxis_title="Score"
            )
        else:
            threshold_fig = go.Figure()
            threshold_fig.update_layout(template="plotly_white")

        table_cols = [
            "row_id",
            "Pass_Fail_Label",
            "model_prediction",
            "fail_probability",
            "error_type",
            *selected_features,
        ]
        table_cols = list(dict.fromkeys(table_cols))
        table_df = filtered[table_cols].head(500).copy()
        numeric_cols = table_df.select_dtypes(include="number").columns
        table_df[numeric_cols] = table_df[numeric_cols].round(4)
        table_columns = [{"name": c, "id": c} for c in table_cols]

        return box_fig, hist_fig, cm_fig, importance_fig, scatter_fig, threshold_fig, table_df.to_dict("records"), table_columns

    @app.callback(
        Output("download-detail-csv", "data"),
        Input("btn-download-csv", "n_clicks"),
        State("filter-actual-class", "value"),
        State("filter-predicted-class", "value"),
        State("filter-error-type", "value"),
        State("filter-fail-probability", "value"),
        prevent_initial_call=True,
    )
    def _download_csv(n_clicks, actual_class, predicted_class, error_type, prob_range):
        filtered = _filtered_df(actual_class, predicted_class, error_type, prob_range)
        return dcc.send_data_frame(filtered.to_csv, "secom_dashboard_filtered.csv", index=False)

    return app


app = _create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
