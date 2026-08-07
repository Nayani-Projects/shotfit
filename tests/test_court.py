from __future__ import annotations

import pandas as pd

from shotfit.court import shot_translation_court


def test_court_renders_volume_and_efficiency_hexes() -> None:
    bins = pd.DataFrame(
        {
            "x_bin": [-2.5, 2.5],
            "y_bin": [0.0, 23.75],
            "attempts": [20, 5],
            "actual_makes": [12, 1],
            "expected_makes": [9.0, 2.0],
            "extra_makes_per_100": [8.0, -5.0],
        }
    )
    figure = shot_translation_court(bins)
    assert figure.data[0].marker.symbol == "hexagon"
    assert figure.data[0].marker.size[0] > figure.data[0].marker.size[1]
    assert "%{customdata[3]:+.1f}" in figure.data[0].hovertemplate
    assert figure.data[0].customdata[0][3] == 8.0
    assert figure.layout.yaxis.scaleanchor == "x"
