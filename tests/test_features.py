from __future__ import annotations

import pandas as pd

from shotfit.features import action_group, shot_family, transform_shots


def test_action_group_normalizes_shot_descriptions() -> None:
    assert action_group("Running Pull-Up Jump Shot") == "pull_up_or_step_back"
    assert action_group("Step Back Jump shot") == "pull_up_or_step_back"
    assert action_group("Driving Finger Roll Layup Shot") == "layup"


def test_shot_family_boundaries() -> None:
    assert shot_family(pd.Series({"shot_distance": 3, "shot_type": "2PT Field Goal", "shot_zone_basic": "Restricted Area"})) == "At the rim"
    assert shot_family(pd.Series({"shot_distance": 23, "shot_type": "3PT Field Goal", "shot_zone_basic": "Left Corner 3"})) == "Corner three"
    assert shot_family(pd.Series({"shot_distance": 26, "shot_type": "3PT Field Goal", "shot_zone_basic": "Above the Break 3"})) == "Above the break"
    assert shot_family(pd.Series({"shot_distance": 12, "shot_type": "2PT Field Goal", "shot_zone_basic": "Mid-Range"})) == "Midrange"


def test_transform_shots_builds_model_fields() -> None:
    row = {"GAME_ID": "22400001", "GAME_EVENT_ID": 12, "PLAYER_ID": 1, "PLAYER_NAME": "A Player", "TEAM_ID": 1610612744, "TEAM_NAME": "Golden State Warriors", "PERIOD": 1, "MINUTES_REMAINING": 0, "SECONDS_REMAINING": 4, "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "3PT Field Goal", "SHOT_ZONE_BASIC": "Left Corner 3", "SHOT_ZONE_AREA": "Left Side(L)", "SHOT_ZONE_RANGE": "24+ ft.", "SHOT_DISTANCE": 23, "LOC_X": -220, "LOC_Y": 10, "SHOT_MADE_FLAG": 1, "GAME_DATE": "20241023", "HTM": "POR", "VTM": "GSW"}
    out = transform_shots(pd.DataFrame([row]), "2024-25").iloc[0]
    assert out.shot_id == "0022400001-12"
    assert out.is_corner_three == 1 and out.shot_value == 3
    assert out.is_late_period == 1 and out.is_home == 0

