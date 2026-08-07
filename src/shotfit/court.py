"""Basketball-native half-court visualization for player shot translation."""

from __future__ import annotations

import math

import plotly.graph_objects as go

ZONE_LAYOUT = {
    "At the rim": {"value": "court_rim_extra", "attempts": "court_rim_attempts", "x": 0, "y": 5.2},
    "Midrange": {"value": "court_midrange_extra", "attempts": "court_midrange_attempts", "x": 0, "y": 18},
    "Left corner": {"value": "left_corner_extra", "attempts": "left_corner_attempts", "x": -23.3, "y": 7},
    "Right corner": {"value": "right_corner_extra", "attempts": "right_corner_attempts", "x": 23.3, "y": 7},
    "Above the break": {"value": "court_above_break_extra", "attempts": "court_above_break_attempts", "x": 0, "y": 33},
}


def _fill(value: float) -> str:
    opacity = min(0.38, 0.12 + abs(value) / 20)
    color = "29, 66, 138" if value >= 0 else "190, 88, 70"
    return f"rgba({color}, {opacity:.3f})"


def _polygon_circle(cx: float, cy: float, radius: float, points: int = 40) -> tuple[list[float], list[float]]:
    angles = [2 * math.pi * index / points for index in range(points + 1)]
    return [cx + radius * math.cos(angle) for angle in angles], [cy + radius * math.sin(angle) for angle in angles]


def shot_translation_court(row) -> go.Figure:
    """Return a directly labeled half-court with five reliability-adjusted zones."""
    figure = go.Figure()
    zone_polygons = {
        "Above the break": ([-21.8, 21.8, 21.8, -21.8, -21.8], [24, 24, 43.5, 43.5, 24]),
        "Midrange": ([-21.8, 21.8, 21.8, -21.8, -21.8], [10.5, 10.5, 23.5, 23.5, 10.5]),
        "Left corner": ([-24.8, -22.1, -22.1, -24.8, -24.8], [0.2, 0.2, 14, 14, 0.2]),
        "Right corner": ([22.1, 24.8, 24.8, 22.1, 22.1], [0.2, 0.2, 14, 14, 0.2]),
        "At the rim": _polygon_circle(0, 5.2, 5.0),
    }
    for zone, (x_values, y_values) in zone_polygons.items():
        value = float(row[ZONE_LAYOUT[zone]["value"]])
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                fill="toself",
                fillcolor=_fill(value),
                line={"color": "rgba(91, 101, 116, .35)", "width": 1},
                hovertemplate=(
                    f"<b>{zone}</b><br>{value:+.1f} extra makes per 100"
                    f"<br>{int(row[ZONE_LAYOUT[zone]['attempts']]):,} shots<extra></extra>"
                ),
                showlegend=False,
            )
        )
    line_color = "rgba(72, 81, 94, .65)"
    figure.update_layout(
        shapes=[
            {"type": "rect", "x0": -25, "x1": 25, "y0": 0, "y1": 47, "line": {"color": line_color, "width": 2}},
            {"type": "rect", "x0": -8, "x1": 8, "y0": 0, "y1": 19, "line": {"color": line_color, "width": 1}},
            {"type": "circle", "x0": -6, "x1": 6, "y0": 13, "y1": 25, "line": {"color": line_color, "width": 1}},
            {"type": "circle", "x0": -0.75, "x1": 0.75, "y0": 4.45, "y1": 5.95, "line": {"color": line_color, "width": 2}},
            {"type": "line", "x0": -3, "x1": 3, "y0": 4, "y1": 4, "line": {"color": line_color, "width": 2}},
            {"type": "line", "x0": -22, "x1": -22, "y0": 0, "y1": 14, "line": {"color": line_color, "width": 1}},
            {"type": "line", "x0": 22, "x1": 22, "y0": 0, "y1": 14, "line": {"color": line_color, "width": 1}},
            {"type": "path", "path": "M -22 14 C -20 29, -11 35, 0 35 C 11 35, 20 29, 22 14", "line": {"color": line_color, "width": 1}},
        ],
        annotations=[
            {
                "x": settings["x"],
                "y": settings["y"],
                "text": (
                    f"<b>{zone.upper()}</b><br>"
                    f"<span style='font-size:17px'>{float(row[settings['value']]):+.1f}</span><br>"
                    f"{int(row[settings['attempts']]):,} shots"
                ),
                "showarrow": False,
                "font": {"size": 11, "color": "#263244"},
                "textangle": -90 if "corner" in zone.lower() else 0,
            }
            for zone, settings in ZONE_LAYOUT.items()
        ],
        xaxis={"range": [-26, 26], "visible": False, "fixedrange": True},
        yaxis={"range": [-1, 48], "visible": False, "fixedrange": True, "scaleanchor": "x", "scaleratio": 1},
        height=560,
        margin={"l": 4, "r": 4, "t": 8, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "white", "font": {"color": "#263244"}},
    )
    return figure
