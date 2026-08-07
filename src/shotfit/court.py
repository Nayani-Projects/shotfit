"""Basketball-native shot-location visualization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _arc(cx: float, cy: float, radius: float, start: float, stop: float, points: int = 100):
    angles = np.linspace(math.radians(start), math.radians(stop), points)
    return cx + radius * np.cos(angles), cy + radius * np.sin(angles)


def _line_trace(x, y, width: float = 1.35) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": "#647083", "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def shot_translation_court(bins: pd.DataFrame) -> go.Figure:
    """Render volume-sized hexes colored by makes above/below expectation."""
    bins = bins[bins.y_bin <= 35].copy()
    figure = go.Figure()
    max_attempts = max(float(bins.attempts.max()), 1.0)
    sizes = 7 + 24 * np.power(bins.attempts / max_attempts, 0.6)
    figure.add_trace(
        go.Scatter(
            x=bins.x_bin,
            y=bins.y_bin,
            mode="markers",
            marker={
                "symbol": "hexagon",
                "size": sizes,
                "color": bins.extra_makes_per_100.clip(-10, 10),
                "cmin": -10,
                "cmax": 10,
                "cmid": 0,
                "colorscale": [[0, "#B65345"], [0.5, "#E9EDF3"], [1, "#1D428A"]],
                "line": {"color": "rgba(255,255,255,.8)", "width": 0.8},
                "colorbar": {
                    "title": {"text": "Extra makes<br>per 100", "side": "right"},
                    "thickness": 12,
                    "len": 0.44,
                    "x": 1.01,
                    "tickvals": [-10, 0, 10],
                    "ticktext": ["−10", "0", "+10"],
                    "outlinewidth": 0,
                },
            },
            customdata=np.column_stack([bins.attempts, bins.actual_makes, bins.expected_makes]),
            hovertemplate=(
                "<b>%{customdata[0]:,.0f} shots</b><br>"
                "%{marker.color:+.1f} extra makes per 100<br>"
                "%{customdata[1]:.0f} actual · %{customdata[2]:.1f} expected<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Regulation NBA half-court geometry in feet, hoop centered at (0, 0).
    figure.add_trace(_line_trace([-25, -25], [-4.75, 35.5], 1.7))
    figure.add_trace(_line_trace([25, 25], [-4.75, 35.5], 1.7))
    figure.add_trace(_line_trace([-25, 25], [-4.75, -4.75], 1.7))
    figure.add_trace(_line_trace([-8, 8, 8, -8, -8], [-4.75, -4.75, 14.25, 14.25, -4.75]))
    x, y = _arc(0, 14.25, 6, 0, 360)
    figure.add_trace(_line_trace(x, y))
    x, y = _arc(0, 0, 4, 0, 180)
    figure.add_trace(_line_trace(x, y))
    x, y = _arc(0, 0, 0.75, 0, 360)
    figure.add_trace(_line_trace(x, y, 1.8))
    figure.add_trace(_line_trace([-3, 3], [-1.25, -1.25], 1.8))
    figure.add_trace(_line_trace([-22, -22], [-4.75, 9.25]))
    figure.add_trace(_line_trace([22, 22], [-4.75, 9.25]))
    angle = math.degrees(math.acos(22 / 23.75))
    x, y = _arc(0, 0, 23.75, angle, 180 - angle)
    figure.add_trace(_line_trace(x, y))

    figure.update_layout(
        xaxis={"range": [-27, 28], "visible": False, "fixedrange": True},
        yaxis={"range": [-5.5, 35.5], "visible": False, "fixedrange": True, "scaleanchor": "x", "scaleratio": 1},
        height=500,
        margin={"l": 8, "r": 78, "t": 4, "b": 4},
        paper_bgcolor="#F7F8FA",
        plot_bgcolor="#F7F8FA",
        hoverlabel={"bgcolor": "white", "font": {"color": "#172033"}},
    )
    return figure
