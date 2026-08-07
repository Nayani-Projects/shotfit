from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_public_bundle_follows_evidence_contract() -> None:
    briefs = pd.read_parquet(ROOT / "data/app/player_briefs.parquet")
    metadata = json.loads((ROOT / "data/app/model_metadata.json").read_text())
    assert metadata["public_estimate_seasons"] == ["2025-26"]
    assert briefs.attempts.min() >= 250
    assert "role" not in briefs.columns
    assert briefs.evidence_label.notna().all()
    assert briefs.teams_represented.ge(1).all()
    assert briefs.team_breakdown.str.len().gt(0).all()
    assert briefs.loc[briefs.evidence_label == "Strong positive evidence", "lower_80"].gt(0).all()
    assert briefs.loc[briefs.evidence_label == "Strong negative evidence", "upper_80"].lt(0).all()
    inconclusive = briefs[briefs.evidence_label == "Inconclusive evidence"]
    assert (inconclusive.lower_80.le(0) & inconclusive.upper_80.ge(0)).all()


def test_court_and_area_artifacts_reconcile_to_qualified_players() -> None:
    briefs = pd.read_parquet(ROOT / "data/app/player_briefs.parquet")
    areas = pd.read_parquet(ROOT / "data/app/player_areas.parquet")
    bins = pd.read_parquet(ROOT / "data/app/player_hex_bins.parquet")
    ids = set(briefs.player_id)
    assert set(areas.player_id) == ids
    assert set(bins.player_id) == ids
    area_attempts = areas.groupby("player_id").attempts.sum()
    brief_attempts = briefs.set_index("player_id").attempts
    pd.testing.assert_series_equal(
        area_attempts.sort_index(), brief_attempts.sort_index(), check_names=False
    )
