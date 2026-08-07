from __future__ import annotations

import pytest

from shotfit.ingest import response_frame, validate_manifest


def test_response_frame_extracts_shot_dataset() -> None:
    payload = {"resultSets": [{"name": "Shot_Chart_Detail", "headers": ["GAME_ID", "GAME_EVENT_ID"], "rowSet": [["1", 2]]}]}
    assert response_frame(payload).to_dict(orient="records") == [{"GAME_ID": "1", "GAME_EVENT_ID": 2}]


def test_response_frame_rejects_missing_dataset() -> None:
    with pytest.raises(ValueError, match="missing"):
        response_frame({"resultSets": []})


def test_manifest_rejects_silent_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_teams = [{"abbreviation": f"T{i}"} for i in range(30)]
    monkeypatch.setattr("shotfit.ingest.teams.get_teams", lambda: fake_teams)
    rows = 102_400 // 30
    manifest = [{"team": f"T{i}", "game_count": 82, "first_game_date": "20241022", "last_game_date": "20250413", "row_count": rows} for i in range(30)]
    manifest[-1]["row_count"] += 102_400 - rows * 30
    with pytest.raises(ValueError, match="suspicious"):
        validate_manifest(manifest, "2024-25")

