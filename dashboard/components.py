"""Reusable Dash UI building blocks for the SECOM analytics dashboard.

Kept deliberately dependency-light (inline styles only, light theme) so the
dashboard has no external CSS/JS requirements beyond Dash itself.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

COLORS = {
    "background": "#F7F8FA",
    "card_background": "#FFFFFF",
    "border": "#E0E3E8",
    "text_primary": "#1F2430",
    "text_secondary": "#5B6472",
    "accent": "#2F6FED",
    "pass_color": "#2E9E5B",
    "fail_color": "#D64545",
    "banner_background": "#FFF4E5",
    "banner_border": "#F0B429",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_background"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "padding": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}


def build_disclaimer_banner() -> html.Div:
    """Build the top-of-page banner stating the dataset's scope and limitations.

    Returns:
        A ``html.Div`` with three explicit disclaimer lines, always shown
        above the fold on every page of the dashboard.
    """
    return html.Div(
        [
            html.Div("Public anonymized SECOM dataset", style={"fontWeight": 700}),
            html.Div(
                "Sensor IDs are anonymized; this dashboard does not identify actual "
                "Fab equipment or process modules."
            ),
            html.Div("No wafer, lot, chamber, or spatial die-coordinate data are available."),
        ],
        style={
            "backgroundColor": COLORS["banner_background"],
            "border": f"1px solid {COLORS['banner_border']}",
            "borderRadius": "8px",
            "padding": "10px 16px",
            "marginBottom": "16px",
            "color": "#7A4A00",
            "fontSize": "13px",
            "lineHeight": "1.5",
        },
    )


def build_kpi_card(label: str, value: str, card_id: str | None = None) -> html.Div:
    """Build a single KPI stat card.

    Args:
        label: Short metric name shown above the value.
        value: Pre-formatted display value (formatting is the caller's job
            so every KPI card uses a consistent number format).
        card_id: Optional Dash component id, so callbacks can update ``value``.

    Returns:
        A styled ``html.Div`` KPI card.
    """
    value_div_kwargs = {"id": card_id} if card_id is not None else {}
    return html.Div(
        [
            html.Div(
                label,
                style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginBottom": "4px"},
            ),
            html.Div(
                value,
                style={"fontSize": "22px", "fontWeight": 700, "color": COLORS["text_primary"]},
                **value_div_kwargs,
            ),
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "140px"},
    )


def build_kpi_row(cards: list[html.Div]) -> html.Div:
    """Lay out a list of KPI cards in a single responsive row.

    Args:
        cards: KPI cards, typically from :func:`build_kpi_card`.

    Returns:
        A flexbox row ``html.Div`` wrapping the cards.
    """
    return html.Div(
        cards,
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"},
    )


def build_chart_card(title: str, graph: dcc.Graph) -> html.Div:
    """Wrap a ``dcc.Graph`` in a titled card for consistent chart-area styling.

    Args:
        title: Chart title shown above the graph.
        graph: The ``dcc.Graph`` component to embed.

    Returns:
        A styled ``html.Div`` containing the title and graph.
    """
    return html.Div(
        [
            html.Div(
                title,
                style={"fontSize": "14px", "fontWeight": 600, "marginBottom": "8px", "color": COLORS["text_primary"]},
            ),
            graph,
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px"},
    )


def build_filter_panel(feature_options: list[str], default_features: list[str]) -> html.Div:
    """Build the filter control panel (actual/predicted class, error type, probability, sensors).

    Args:
        feature_options: All selectable ``feature_XXX`` column names.
        default_features: Pre-selected features for the sensor dropdown
            (typically the top-importance features).

    Returns:
        A styled ``html.Div`` containing every filter control, each with a
        stable Dash component id used by the app's callbacks.
    """
    label_style = {"fontSize": "12px", "fontWeight": 600, "color": COLORS["text_secondary"], "marginBottom": "4px"}

    def _field(label: str, control: Any) -> html.Div:
        return html.Div([html.Div(label, style=label_style), control], style={"minWidth": "200px"})

    return html.Div(
        [
            _field(
                "Actual Pass/Fail",
                dcc.Dropdown(
                    id="filter-actual-class",
                    options=[{"label": "All", "value": "ALL"}, {"label": "Pass", "value": "Pass"}, {"label": "Fail", "value": "Fail"}],
                    value="ALL",
                    clearable=False,
                ),
            ),
            _field(
                "Predicted Pass/Fail",
                dcc.Dropdown(
                    id="filter-predicted-class",
                    options=[{"label": "All", "value": "ALL"}, {"label": "Pass", "value": "Pass"}, {"label": "Fail", "value": "Fail"}],
                    value="ALL",
                    clearable=False,
                ),
            ),
            _field(
                "Error Type",
                dcc.Dropdown(
                    id="filter-error-type",
                    options=[{"label": t, "value": t} for t in ["All", "TP", "TN", "FP", "FN"]],
                    value="All",
                    clearable=False,
                ),
            ),
            _field(
                "Fail Probability Range",
                dcc.RangeSlider(id="filter-fail-probability", min=0, max=1, step=0.01, value=[0, 1], marks={0: "0", 0.5: "0.5", 1: "1"}),
            ),
            _field(
                "Sensor(s) to Inspect",
                dcc.Dropdown(
                    id="filter-selected-features",
                    options=[{"label": f, "value": f} for f in feature_options],
                    value=default_features,
                    multi=True,
                ),
            ),
        ],
        style={
            **CARD_STYLE,
            "display": "flex",
            "gap": "20px",
            "flexWrap": "wrap",
            "marginBottom": "20px",
            "alignItems": "flex-start",
        },
    )
