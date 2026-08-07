"""Shared configuration and filesystem locations."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "nba_stats" / "shotchartdetail"
DB_PATH = DATA_DIR / "database" / "shotfit.duckdb"
PROCESSED_DIR = DATA_DIR / "processed"
APP_DATA_DIR = DATA_DIR / "app"
REFERENCE_DIR = DATA_DIR / "reference"
ARTIFACTS_DIR = ROOT / "artifacts"

SEASONS = ("2022-23", "2023-24", "2024-25")
TRAIN_SEASON = "2022-23"
VALIDATION_SEASON = "2023-24"
TEST_SEASON = "2024-25"
MIN_TEST_ATTEMPTS = 250

EXPECTED_DATES = {
    "2022-23": ("20221018", "20230409"),
    "2023-24": ("20231024", "20240414"),
    "2024-25": ("20241022", "20250413"),
}

REQUIRED_COLUMNS = (
    "GAME_ID",
    "GAME_EVENT_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_NAME",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "ACTION_TYPE",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "SHOT_MADE_FLAG",
    "GAME_DATE",
    "HTM",
    "VTM",
)
