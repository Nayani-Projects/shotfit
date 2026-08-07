"""Resumable, validated ShotChartDetail ingestion."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import nba_api
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

from shotfit.config import EXPECTED_DATES, RAW_DIR, REQUIRED_COLUMNS, SEASONS


def _dataset(payload: dict) -> tuple[list[str], list[list]]:
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for result in result_sets or []:
        name = str(result.get("name", "")).lower().replace("_", "")
        if name == "shotchartdetail":
            return result["headers"], result["rowSet"]
    raise ValueError("Shot_Chart_Detail dataset missing from API response")


def response_frame(payload: dict) -> pd.DataFrame:
    """Convert a raw NBA Stats payload into its shot-detail frame."""
    headers, rows = _dataset(payload)
    return pd.DataFrame(rows, columns=headers)


def cache_paths(season: str, abbreviation: str, raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    base = raw_dir / season / abbreviation
    return base.with_suffix(".json.gz"), base.with_suffix(".metadata.json")


def valid_cache(raw_path: Path, metadata_path: Path) -> bool:
    """Require readable gzip content whose uncompressed payload matches metadata."""
    if not raw_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
            raw_text = stream.read()
        if hashlib.sha256(raw_text.encode()).hexdigest() != metadata["content_sha256"]:
            return False
        response_frame(json.loads(raw_text))
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def ingest_season(
    season: str,
    *,
    raw_dir: Path = RAW_DIR,
    refresh: bool = False,
    min_interval: float = 0.15,
    timeout: int = 45,
) -> list[dict]:
    """Fetch one team-sized response at a time and preserve raw API JSON."""
    if season not in EXPECTED_DATES:
        raise ValueError(f"Unsupported season: {season}")
    manifest: list[dict] = []
    last_request = 0.0
    for team in teams.get_teams():
        abbreviation = team["abbreviation"]
        raw_path, metadata_path = cache_paths(season, abbreviation, raw_dir)
        if valid_cache(raw_path, metadata_path) and not refresh:
            manifest.append(json.loads(metadata_path.read_text()))
            continue
        wait = min_interval - (time.monotonic() - last_request)
        if wait > 0:
            time.sleep(wait)
        endpoint = shotchartdetail.ShotChartDetail(
            team_id=team["id"],
            player_id=0,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
            timeout=timeout,
        )
        last_request = time.monotonic()
        raw_text = endpoint.get_json()
        payload = json.loads(raw_text)
        frame = response_frame(payload)
        _validate_team_frame(frame, season, abbreviation)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
            stream.write(raw_text)
        metadata = {
            "endpoint": "shotchartdetail",
            "season": season,
            "season_type": "Regular Season",
            "team_id": int(team["id"]),
            "team": abbreviation,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "nba_api_version": getattr(nba_api, "__version__", "unknown"),
            "row_count": int(len(frame)),
            "game_count": int(frame["GAME_ID"].nunique()),
            "first_game_date": str(frame["GAME_DATE"].min()),
            "last_game_date": str(frame["GAME_DATE"].max()),
            "schema_version": 1,
            "content_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(metadata)
    validate_manifest(manifest, season)
    manifest_path = raw_dir / season / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def ingest_all(seasons: tuple[str, ...] = SEASONS, **kwargs) -> dict[str, list[dict]]:
    return {season: ingest_season(season, **kwargs) for season in seasons}


def _validate_team_frame(frame: pd.DataFrame, season: str, team: str) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{season} {team}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{season} {team}: empty response")
    if frame.duplicated(["GAME_ID", "GAME_EVENT_ID"]).any():
        raise ValueError(f"{season} {team}: duplicate game-event identifiers")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{season} {team}: required fields contain nulls")


def validate_manifest(manifest: list[dict], season: str) -> None:
    expected_teams = {team["abbreviation"] for team in teams.get_teams()}
    actual_teams = {item["team"] for item in manifest}
    if actual_teams != expected_teams:
        raise ValueError(f"{season}: missing teams {sorted(expected_teams - actual_teams)}")
    if any(item["game_count"] != 82 for item in manifest):
        bad = [item["team"] for item in manifest if item["game_count"] != 82]
        raise ValueError(f"{season}: incomplete team schedules {bad}")
    first, last = EXPECTED_DATES[season]
    if min(item["first_game_date"] for item in manifest) != first:
        raise ValueError(f"{season}: unexpected opening date")
    if max(item["last_game_date"] for item in manifest) != last:
        raise ValueError(f"{season}: unexpected closing date")
    total = sum(item["row_count"] for item in manifest)
    if total == 102_400 or total < 180_000:
        raise ValueError(f"{season}: suspicious row count {total}")
