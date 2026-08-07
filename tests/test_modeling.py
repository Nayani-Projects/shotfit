from __future__ import annotations

import numpy as np
import pandas as pd

from shotfit.evaluation import empirical_bayes, evidence_label
from shotfit.modeling import choose_model, expected_calibration_error


def test_calibration_error_is_zero_for_calibrated_groups() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, bins=2) == 0


def test_model_selection_prefers_simplicity_without_material_gain() -> None:
    results = {"Logistic regression": {"log_loss": 0.60, "brier_score": 0.20}, "XGBoost": {"log_loss": 0.598, "brier_score": 0.1995}}
    assert choose_model(results) == "Logistic regression"


def test_empirical_bayes_shrinks_small_samples_more() -> None:
    rows = []
    for player, attempts in ((1, 20), (2, 1000)):
        for i in range(attempts):
            rows.append({"player_id": player, "shot_made": int(i % 2 == 0), "predicted_make_probability": 0.4})
    out = empirical_bayes(pd.DataFrame(rows), ["player_id"]).set_index("player_id")
    assert out.loc[1, "shrinkage_weight"] < out.loc[2, "shrinkage_weight"]
    assert (out.loc[1, "upper_80"] - out.loc[1, "lower_80"]) > (out.loc[2, "upper_80"] - out.loc[2, "lower_80"])


def test_evidence_labels_follow_interval_bounds() -> None:
    intervals = pd.DataFrame(
        {"lower_80": [0.1, -1.0, -3.0], "upper_80": [2.0, 2.0, -0.1]}
    )
    assert evidence_label(intervals).tolist() == [
        "Strong positive evidence",
        "Inconclusive evidence",
        "Strong negative evidence",
    ]
