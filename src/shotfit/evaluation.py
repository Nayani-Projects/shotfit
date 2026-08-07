"""Empirical-Bayes player summaries, roles, and monitoring outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shotfit.config import (
    APP_DATA_DIR,
    ARTIFACTS_DIR,
    MIN_TEST_ATTEMPTS,
    PROCESSED_DIR,
    REFERENCE_DIR,
    TEST_SEASON,
)
from shotfit.rosters import POSITION_OVERRIDES

FAMILIES = ("At the rim", "Midrange", "Corner three", "Above the break")
FAMILY_SLUG = {"At the rim": "rim", "Midrange": "midrange", "Corner three": "corner_three", "Above the break": "above_break"}


def empirical_bayes(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns).agg(
        attempts=("shot_made", "size"),
        actual_makes=("shot_made", "sum"),
        expected_makes=("predicted_make_probability", "sum"),
        bernoulli_variance=("predicted_make_probability", lambda p: float((p * (1 - p)).sum())),
    ).reset_index()
    grouped["observed_rate"] = (grouped.actual_makes - grouped.expected_makes) / grouped.attempts
    grouped["sampling_variance"] = grouped.bernoulli_variance / grouped.attempts.pow(2)
    # Weight the population moment by attempts so tiny, noisy player-family
    # groups do not collapse the estimated between-player signal toward zero.
    population_mean = float(np.average(grouped.observed_rate, weights=grouped.attempts))
    population_variance = float(
        np.average((grouped.observed_rate - population_mean) ** 2, weights=grouped.attempts)
    )
    sampling_variance = float(np.average(grouped.sampling_variance, weights=grouped.attempts))
    prior_variance = max(population_variance - sampling_variance, 1e-4)
    grouped["shrinkage_weight"] = prior_variance / (prior_variance + grouped.sampling_variance)
    grouped["posterior_rate"] = grouped.observed_rate * grouped.shrinkage_weight
    grouped["posterior_variance"] = 1 / (1 / prior_variance + 1 / grouped.sampling_variance.clip(lower=1e-12))
    grouped["extra_makes_per_100"] = grouped.posterior_rate * 100
    radius = 1.281552 * np.sqrt(grouped.posterior_variance) * 100
    grouped["lower_80"] = grouped.extra_makes_per_100 - radius
    grouped["upper_80"] = grouped.extra_makes_per_100 + radius
    return grouped


def role_for_player(row: pd.Series) -> tuple[str, str]:
    if row.rim_frequency >= 0.45 and row.rim_extra > 0:
        return "Rim-pressure finisher", "Prioritize paint pressure and finishing opportunities; verify how much creation versus setup drives the volume."
    if row.creator_frequency >= 0.25 and row.extra_makes_per_100 > 0:
        return "Secondary shot creator", "Use selective pull-up and step-back creation while preserving a strong share of assisted opportunities."
    if row.three_frequency >= 0.55 and (row.corner_three_extra + row.above_break_extra) / 2 > 0:
        return "Perimeter spacer", "Prioritize corner and above-the-break threes; use film to verify movement and release-window demands."
    return "Balanced scoring wing", "Investigate a mixed finishing and perimeter role rather than concentrating volume in one shot family."


def _confidence(row: pd.Series) -> str:
    width = float(row.upper_80 - row.lower_80)
    if row.attempts >= 800 and width <= 4:
        return "High confidence"
    if row.attempts >= 400 and width <= 7:
        return "Moderate confidence"
    return "Low confidence"


def _bottom_line(row: pd.Series) -> str:
    value = float(row.extra_makes_per_100)
    if row.lower_80 > 0:
        return "The shooting signal looks worth deeper investigation."
    if row.upper_80 < 0:
        return "The public shot results raise a translation concern."
    if value >= 1:
        return "There is a positive signal, but the range remains uncertain."
    if value <= -1:
        return "The results trail expectation, but the range remains uncertain."
    return "The public evidence is inconclusive."


def _strongest(row: pd.Series) -> str:
    values = {family: float(row[f"{FAMILY_SLUG[family]}_extra"]) for family in FAMILIES}
    family = max(values, key=values.get)
    return f"The strongest positive evidence comes from {family.lower()} attempts ({values[family]:+.1f} extra makes per 100)."


def build_player_briefs(scored: pd.DataFrame) -> pd.DataFrame:
    overall = empirical_bayes(scored, ["player_id", "player_name"])
    family = empirical_bayes(scored, ["player_id", "player_name", "shot_family"])
    family_wide = family.pivot(index=["player_id", "player_name"], columns="shot_family", values="extra_makes_per_100").reset_index()
    family_attempts = family.pivot(index=["player_id", "player_name"], columns="shot_family", values="attempts").reset_index()
    for family_name in FAMILIES:
        if family_name not in family_wide:
            family_wide[family_name] = 0.0
        if family_name not in family_attempts:
            family_attempts[family_name] = 0
    family_wide[list(FAMILIES)] = family_wide[list(FAMILIES)].fillna(0.0)
    family_attempts[list(FAMILIES)] = family_attempts[list(FAMILIES)].fillna(0).astype(int)
    family_wide = family_wide.rename(columns={family: f"{FAMILY_SLUG[family]}_extra" for family in FAMILIES})
    family_attempts = family_attempts.rename(columns={family: f"{FAMILY_SLUG[family]}_attempts" for family in FAMILIES})
    briefs = overall.merge(family_wide, on=["player_id", "player_name"]).merge(family_attempts, on=["player_id", "player_name"])
    test_counts = scored[scored.season == TEST_SEASON].groupby("player_id").size().rename("test_attempts")
    seasons = scored.groupby("player_id").season.nunique().rename("seasons_reviewed")
    creator = scored.assign(is_creator=scored.action_group.eq("pull_up_or_step_back")).groupby("player_id").is_creator.mean().rename("creator_frequency")
    rim = scored.assign(is_rim=scored.shot_family.eq("At the rim")).groupby("player_id").is_rim.mean().rename("rim_frequency")
    threes = scored.assign(is_three=scored.shot_value.eq(3)).groupby("player_id").is_three.mean().rename("three_frequency")
    briefs = briefs.merge(test_counts, on="player_id").merge(seasons, on="player_id").merge(creator, on="player_id").merge(rim, on="player_id").merge(threes, on="player_id")
    briefs = briefs[briefs.test_attempts >= MIN_TEST_ATTEMPTS].copy()
    test_shots = scored[scored.season == TEST_SEASON].sort_values(["game_date", "game_id", "game_event_id"])
    latest_team = test_shots.drop_duplicates("player_id", keep="last")[["player_id", "team_id", "team_name"]]
    positions_path = REFERENCE_DIR / f"player_positions_{TEST_SEASON.replace('-', '_')}.parquet"
    if not positions_path.exists():
        raise FileNotFoundError("Player positions are missing. Run `python -m shotfit.cli fetch-rosters` first.")
    positions = pd.read_parquet(positions_path)[["player_id", "position"]]
    briefs = briefs.merge(latest_team, on="player_id", how="left").merge(positions, on="player_id", how="left")
    briefs["position"] = briefs.position.fillna(briefs.player_id.map(POSITION_OVERRIDES)).fillna("Not listed")
    roles = briefs.apply(role_for_player, axis=1)
    briefs["role"] = [item[0] for item in roles]
    briefs["role_description"] = [item[1] for item in roles]
    briefs["confidence"] = briefs.apply(_confidence, axis=1)
    briefs["bottom_line"] = briefs.apply(_bottom_line, axis=1)
    briefs["strongest_evidence"] = briefs.apply(_strongest, axis=1)
    briefs["positive_families"] = briefs[[f"{FAMILY_SLUG[f]}_extra" for f in FAMILIES]].gt(0).sum(axis=1)
    briefs["repeat_label"] = np.select([briefs.positive_families >= 3, briefs.positive_families == 2], ["Yes", "Mostly"], default="No")
    keep = [
        "player_id", "player_name", "team_id", "team_name", "position", "attempts", "test_attempts", "seasons_reviewed", "extra_makes_per_100", "lower_80", "upper_80", "confidence", "bottom_line", "positive_families", "repeat_label", "role", "role_description", "strongest_evidence",
        *[f"{FAMILY_SLUG[f]}_extra" for f in FAMILIES],
        *[f"{FAMILY_SLUG[f]}_attempts" for f in FAMILIES],
    ]
    return briefs[keep].sort_values("extra_makes_per_100", ascending=False).reset_index(drop=True)


def monitoring_report(scored: pd.DataFrame) -> dict:
    validation = scored[scored.split == "validation"]
    test = scored[scored.split == "test"]
    numeric = ["shot_distance", "loc_x", "loc_y", "shot_angle", "seconds_remaining_period"]
    drift = []
    for column in numeric:
        pooled = float(pd.concat([validation[column], test[column]]).std()) or 1.0
        shift = abs(float(test[column].mean() - validation[column].mean())) / pooled
        drift.append({"feature": column, "standardized_mean_shift": round(shift, 4), "status": "review" if shift >= 0.2 else "stable"})
    prediction_shift = abs(float(test.predicted_make_probability.mean() - validation.predicted_make_probability.mean()))
    status = "review" if prediction_shift >= 0.03 or any(item["status"] == "review" for item in drift) else "stable"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "thresholds": {"numeric_standardized_mean_shift": 0.2, "mean_prediction_shift": 0.03},
        "missing_cells": {"validation": int(validation.isna().sum().sum()), "test": int(test.isna().sum().sum())},
        "mean_prediction": {"validation": float(validation.predicted_make_probability.mean()), "test": float(test.predicted_make_probability.mean()), "absolute_shift": prediction_shift},
        "feature_drift": drift,
    }


def export_app_bundle(
    prediction_path: Path = PROCESSED_DIR / "model_predictions.parquet",
    app_dir: Path = APP_DATA_DIR,
    metrics_path: Path = ARTIFACTS_DIR / "metrics.json",
) -> tuple[pd.DataFrame, dict]:
    scored = pd.read_parquet(prediction_path)
    briefs = build_player_briefs(scored)
    monitoring = monitoring_report(scored)
    metrics = json.loads(metrics_path.read_text())
    app_dir.mkdir(parents=True, exist_ok=True)
    briefs.to_parquet(app_dir / "player_briefs.parquet", index=False)
    (app_dir / "validation_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (app_dir / "monitoring.json").write_text(json.dumps(monitoring, indent=2) + "\n")
    metadata = {"model_version": metrics["model_version"], "created_at": metrics["created_at"], "qualified_players": int(len(briefs)), "minimum_test_attempts": MIN_TEST_ATTEMPTS, "evaluation_season": TEST_SEASON, "supporting_season": metrics["seasons"]["validation"], "runtime_network_calls": False}
    (app_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return briefs, monitoring
