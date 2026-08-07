"""Normalize cached shots and build leakage-safe model features."""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import duckdb
import pandas as pd
from nba_api.stats.static import teams

from shotfit.config import DB_PATH, PROCESSED_DIR, RAW_DIR, REQUIRED_COLUMNS, SEASONS
from shotfit.ingest import response_frame

RENAME = {column: column.lower() for column in REQUIRED_COLUMNS}
MODEL_FEATURES = (
    "shot_distance",
    "loc_x",
    "loc_y",
    "shot_angle",
    "shot_value",
    "is_corner_three",
    "period",
    "seconds_remaining_period",
    "is_late_period",
    "is_home",
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "action_group",
    "shot_family",
)
NUMERIC_FEATURES = (
    "shot_distance",
    "loc_x",
    "loc_y",
    "shot_angle",
    "shot_value",
    "is_corner_three",
    "period",
    "seconds_remaining_period",
    "is_late_period",
    "is_home",
)
CATEGORICAL_FEATURES = (
    "shot_zone_basic",
    "shot_zone_area",
    "shot_zone_range",
    "action_group",
    "shot_family",
)


def action_group(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    if any(token in text for token in ("pullup", "pull up", "step back", "stepback")):
        return "pull_up_or_step_back"
    if "dunk" in text:
        return "dunk"
    if any(token in text for token in ("layup", "finger roll")):
        return "layup"
    if "hook" in text:
        return "hook"
    if "fadeaway" in text or "fade away" in text:
        return "fadeaway"
    if "jump" in text:
        return "jump_shot"
    if "tip" in text:
        return "tip"
    return "other"


def shot_family(row: pd.Series) -> str:
    distance = float(row["shot_distance"])
    shot_type = str(row["shot_type"])
    zone = str(row["shot_zone_basic"])
    if distance <= 4:
        return "At the rim"
    if "3PT" in shot_type and "Corner" in zone:
        return "Corner three"
    if "3PT" in shot_type:
        return "Above the break"
    return "Midrange"


def transform_shots(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Create one normalized, model-ready row per shot."""
    out = frame[list(REQUIRED_COLUMNS)].rename(columns=RENAME).copy()
    out.insert(0, "season", season)
    out.insert(1, "season_type", "Regular Season")
    out["game_id"] = out["game_id"].astype(str).str.zfill(10)
    out["game_event_id"] = pd.to_numeric(out["game_event_id"]).astype(int)
    out["player_id"] = pd.to_numeric(out["player_id"]).astype(int)
    out["team_id"] = pd.to_numeric(out["team_id"]).astype(int)
    numeric = ["period", "minutes_remaining", "seconds_remaining", "shot_distance", "loc_x", "loc_y", "shot_made_flag"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column])
    out["shot_made"] = out.pop("shot_made_flag").astype(int)
    out["shot_value"] = out["shot_type"].str.contains("3PT").astype(int) + 2
    out["is_corner_three"] = (
        out["shot_type"].str.contains("3PT") & out["shot_zone_basic"].str.contains("Corner")
    ).astype(int)
    out["shot_angle"] = out.apply(lambda row: math.degrees(math.atan2(abs(row.loc_x), max(row.loc_y, 1))), axis=1)
    out["seconds_remaining_period"] = out["minutes_remaining"] * 60 + out["seconds_remaining"]
    out["is_late_period"] = (out["seconds_remaining_period"] <= 5).astype(int)
    abbreviations = {team["id"]: team["abbreviation"] for team in teams.get_teams()}
    out["team_abbreviation"] = out["team_id"].map(abbreviations)
    out["is_home"] = (out["team_abbreviation"] == out["htm"]).astype(int)
    out["action_group"] = out["action_type"].map(action_group)
    out["shot_family"] = out.apply(shot_family, axis=1)
    out["shot_id"] = out["game_id"] + "-" + out["game_event_id"].astype(str)
    return out


def build_database(
    *, raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH, processed_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    schemas: set[tuple[str, ...]] = set()
    for season in SEASONS:
        paths = sorted((raw_dir / season).glob("*.json.gz"))
        if len(paths) != 30:
            raise ValueError(f"{season}: expected 30 cached team responses, found {len(paths)}")
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                raw = response_frame(json.load(stream))
            schemas.add(tuple(raw.columns))
            frames.append(transform_shots(raw, season))
    if len(schemas) != 1:
        raise ValueError("ShotChartDetail schema changed across cached responses")
    shots = pd.concat(frames, ignore_index=True)
    if shots.duplicated(["season_type", "game_id", "game_event_id"]).any():
        raise ValueError("Duplicate game-event identifiers across the normalized dataset")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("shots_df", shots)
        con.execute("CREATE OR REPLACE TABLE raw_shots AS SELECT * FROM shots_df")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS shot_key ON raw_shots(season_type, game_id, game_event_id)")
    finally:
        con.close()
    processed_dir.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(processed_dir / "shot_features.parquet", index=False)
    return shots

