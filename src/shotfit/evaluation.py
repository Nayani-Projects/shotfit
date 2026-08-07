"""Player evidence, validation standards, and app-ready outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

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

COURT_ZONES = ("At the rim", "Midrange", "Left corner three", "Right corner three", "Above the break")
PROFILE_AREAS = ("rim", "midrange", "three")
PROFILE_LABELS = {"rim": "Rim-heavy", "midrange": "Midrange-heavy", "three": "Perimeter-heavy"}
INTERVAL_Z = {"80%": 1.281552, "90%": 1.644854, "95%": 1.959964}
VOLUME_THRESHOLDS = (150, 250, 400, 600)


def _ordinal(value: float) -> str:
    number = int(round(value))
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def empirical_bayes(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns).agg(
        attempts=("shot_made", "size"),
        actual_makes=("shot_made", "sum"),
        expected_makes=("predicted_make_probability", "sum"),
        bernoulli_variance=("predicted_make_probability", lambda p: float((p * (1 - p)).sum())),
    ).reset_index()
    grouped["raw_extra_makes"] = grouped.actual_makes - grouped.expected_makes
    grouped["observed_rate"] = grouped.raw_extra_makes / grouped.attempts
    grouped["sampling_variance"] = grouped.bernoulli_variance / grouped.attempts.pow(2)
    population_mean = float(np.average(grouped.observed_rate, weights=grouped.attempts))
    population_variance = float(np.average((grouped.observed_rate - population_mean) ** 2, weights=grouped.attempts))
    sampling_variance = float(np.average(grouped.sampling_variance, weights=grouped.attempts))
    prior_variance = max(population_variance - sampling_variance, 1e-4)
    grouped["shrinkage_weight"] = prior_variance / (prior_variance + grouped.sampling_variance)
    grouped["posterior_rate"] = grouped.observed_rate * grouped.shrinkage_weight
    grouped["posterior_variance"] = 1 / (1 / prior_variance + 1 / grouped.sampling_variance.clip(lower=1e-12))
    grouped["extra_makes_per_100"] = grouped.posterior_rate * 100
    grouped["adjusted_extra_makes"] = grouped.posterior_rate * grouped.attempts
    posterior_sd = np.sqrt(grouped.posterior_variance)
    grouped["lower_80"] = (grouped.posterior_rate - INTERVAL_Z["80%"] * posterior_sd) * 100
    grouped["upper_80"] = (grouped.posterior_rate + INTERVAL_Z["80%"] * posterior_sd) * 100
    normal = NormalDist()
    grouped["probability_positive"] = [normal.cdf(float(mean / sd)) for mean, sd in zip(grouped.posterior_rate, posterior_sd, strict=True)]
    return grouped


def evidence_label(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [frame.lower_80 > 0, frame.upper_80 < 0],
            ["Strong positive evidence", "Strong negative evidence"],
            default="Inconclusive evidence",
        ),
        index=frame.index,
    )


def _court_zone(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame.shot_family.eq("Corner three") & frame.shot_zone_area.str.contains("Left", case=False, na=False),
                frame.shot_family.eq("Corner three") & frame.shot_zone_area.str.contains("Right", case=False, na=False),
            ],
            ["Left corner three", "Right corner three"],
            default=frame.shot_family,
        ),
        index=frame.index,
    )


def _roster(test: pd.DataFrame) -> pd.DataFrame:
    latest = test.sort_values(["game_date", "game_id", "game_event_id"]).drop_duplicates("player_id", keep="last")
    latest = latest[["player_id", "player_name", "team_id", "team_name"]]
    positions_path = REFERENCE_DIR / f"player_positions_{TEST_SEASON.replace('-', '_')}.parquet"
    if not positions_path.exists():
        raise FileNotFoundError("Player positions are missing. Run `python -m shotfit.cli fetch-rosters` first.")
    positions = pd.read_parquet(positions_path)[["player_id", "position"]]
    roster = latest.merge(positions, on="player_id", how="left")
    roster["position"] = roster.position.fillna(roster.player_id.map(POSITION_OVERRIDES)).fillna("Not listed")
    return roster


def _shot_shares(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby("player_id").agg(
        attempts=("shot_made", "size"),
        rim_share=("shot_family", lambda values: float(values.eq("At the rim").mean())),
        midrange_share=("shot_family", lambda values: float(values.eq("Midrange").mean())),
        three_share=("shot_value", lambda values: float(values.eq(3).mean())),
    ).reset_index()
    return summary


def _profile_percentiles(test_shares: pd.DataFrame, validation_shares: pd.DataFrame) -> pd.DataFrame:
    result = test_shares.copy()
    for area in PROFILE_AREAS:
        column = f"{area}_share"
        output = []
        for row in result.itertuples():
            benchmark = validation_shares.loc[validation_shares.position == row.position, column]
            output.append(float((benchmark <= getattr(row, column)).mean()) if len(benchmark) else 0.5)
        result[f"{area}_percentile"] = output
    percentile_columns = [f"{area}_percentile" for area in PROFILE_AREAS]
    result["profile_area"] = result[percentile_columns].idxmax(axis=1).str.removesuffix("_percentile")
    result["shot_profile"] = np.where(
        result[percentile_columns].max(axis=1) >= 0.75,
        result.profile_area.map(PROFILE_LABELS),
        "Balanced",
    )
    result["profile_percentile"] = result[percentile_columns].max(axis=1)
    result["profile_statement"] = result.apply(
        lambda row: (
            f"{row.shot_profile.replace('-heavy', '')} frequency: "
            f"{_ordinal(row.profile_percentile * 100)} percentile among {row.position.lower()}s"
            if row.shot_profile != "Balanced"
            else "No shot area reached the 75th positional percentile; profile classified as balanced"
        ),
        axis=1,
    )
    return result


def build_area_evidence(test: pd.DataFrame) -> pd.DataFrame:
    areas = empirical_bayes(test.assign(shot_area=_court_zone(test)), ["player_id", "shot_area"])
    totals = areas.groupby("player_id").attempts.transform("sum")
    areas["shot_share"] = areas.attempts / totals
    areas["evidence_label"] = evidence_label(areas)
    areas["is_top_two_volume"] = areas.groupby("player_id").attempts.rank(method="first", ascending=False) <= 2
    return areas[["player_id", "shot_area", "attempts", "shot_share", "actual_makes", "expected_makes", "raw_extra_makes", "adjusted_extra_makes", "extra_makes_per_100", "lower_80", "upper_80", "probability_positive", "evidence_label", "is_top_two_volume"]]


def build_player_briefs(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = scored[scored.split == "test"].copy()
    validation = scored[scored.split == "validation"].copy()
    roster = _roster(test)
    overall = empirical_bayes(test, ["player_id"])
    overall = overall[overall.attempts >= MIN_TEST_ATTEMPTS].copy().merge(roster, on="player_id", how="left")
    overall["evidence_label"] = evidence_label(overall)
    test_shares = _shot_shares(test).merge(overall[["player_id", "position"]], on="player_id")
    eligible_validation = validation[validation.player_id.isin(overall.player_id)]
    validation_shares = _shot_shares(eligible_validation).merge(overall[["player_id", "position"]], on="player_id")
    profiles = _profile_percentiles(test_shares, validation_shares)
    overall = overall.merge(profiles.drop(columns="attempts"), on=["player_id", "position"])
    areas = build_area_evidence(test[test.player_id.isin(overall.player_id)])
    validation_for_areas = validation[validation.player_id.isin(overall.player_id)].copy()
    validation_for_areas["shot_area"] = _court_zone(validation_for_areas)
    validation_area_shares = validation_for_areas.groupby(["player_id", "shot_area"]).size().rename("attempts").reset_index()
    validation_area_shares["shot_share"] = validation_area_shares.attempts / validation_area_shares.groupby("player_id").attempts.transform("sum")
    positions = overall[["player_id", "position"]]
    validation_area_shares = validation_area_shares.merge(positions, on="player_id", how="left")
    area_positions = areas.merge(positions, on="player_id", how="left")
    area_percentiles = []
    for row in area_positions.itertuples():
        benchmark = validation_area_shares.loc[(validation_area_shares.position == row.position) & (validation_area_shares.shot_area == row.shot_area), "shot_share"]
        area_percentiles.append(float((benchmark <= row.shot_share).mean()) if len(benchmark) else 0.5)
    areas["position_frequency_percentile"] = area_percentiles
    strongest = areas[areas.evidence_label == "Strong positive evidence"].sort_values(["player_id", "lower_80"], ascending=[True, False]).drop_duplicates("player_id")
    strongest = strongest[["player_id", "shot_area"]].rename(columns={"shot_area": "strongest_supported_area"})
    flags = areas[areas.is_top_two_volume & areas.evidence_label.eq("Strong negative evidence")].sort_values(["player_id", "upper_80"]).drop_duplicates("player_id")
    flags = flags[["player_id", "shot_area", "shot_share", "extra_makes_per_100"]].rename(columns={"shot_area": "flag_area", "shot_share": "flag_share", "extra_makes_per_100": "flag_extra_per_100"})
    overall = overall.merge(strongest, on="player_id", how="left").merge(flags, on="player_id", how="left")
    overall["strongest_supported_area"] = overall.strongest_supported_area.fillna("No area with conclusive positive evidence")
    overall["review_flag"] = overall.apply(
        lambda row: (
            f"{row.flag_area} accounted for {row.flag_share:.0%} of attempts and had strong negative evidence ({row.flag_extra_per_100:+.1f} adjusted makes per 100)."
            if pd.notna(row.flag_area)
            else "No high-volume area had strong negative evidence."
        ),
        axis=1,
    )
    keep = [
        "player_id", "player_name", "team_id", "team_name", "position", "attempts", "actual_makes", "expected_makes", "raw_extra_makes", "adjusted_extra_makes", "extra_makes_per_100", "lower_80", "upper_80", "probability_positive", "evidence_label", "rim_share", "midrange_share", "three_share", "rim_percentile", "midrange_percentile", "three_percentile", "shot_profile", "profile_statement", "strongest_supported_area", "review_flag",
    ]
    return overall[keep].sort_values("lower_80", ascending=False).reset_index(drop=True), areas


def threshold_analysis(validation: pd.DataFrame) -> list[dict]:
    estimates = empirical_bayes(validation, ["player_id"])
    ordered = validation.sort_values(["player_id", "game_date", "game_id", "game_event_id"]).copy()
    ordered["half"] = ordered.groupby("player_id").cumcount().mod(2)
    split = ordered.groupby(["player_id", "half"]).apply(lambda group: float((group.shot_made - group.predicted_make_probability).mean()), include_groups=False).unstack()
    results = []
    for threshold in VOLUME_THRESHOLDS:
        eligible = estimates[estimates.attempts >= threshold]
        ids = eligible.player_id
        stability = split.loc[split.index.intersection(ids)].dropna().corr().iloc[0, 1] if len(ids) else float("nan")
        labels = evidence_label(eligible)
        results.append(
            {
                "minimum_attempts": threshold,
                "eligible_players": int(len(eligible)),
                "median_interval_width": float((eligible.upper_80 - eligible.lower_80).median()),
                "conclusive_share": float(labels.ne("Inconclusive evidence").mean()),
                "split_half_correlation": float(stability),
                "selected": threshold == MIN_TEST_ATTEMPTS,
            }
        )
    return results


def interval_sensitivity(test: pd.DataFrame) -> list[dict]:
    estimates = empirical_bayes(test, ["player_id"])
    estimates = estimates[estimates.attempts >= MIN_TEST_ATTEMPTS].copy()
    sd = np.sqrt(estimates.posterior_variance) * 100
    output = []
    for label, z_value in INTERVAL_Z.items():
        lower = estimates.extra_makes_per_100 - z_value * sd
        upper = estimates.extra_makes_per_100 + z_value * sd
        output.append(
            {
                "interval": label,
                "strong_positive": int((lower > 0).sum()),
                "inconclusive": int(((lower <= 0) & (upper >= 0)).sum()),
                "strong_negative": int((upper < 0).sum()),
            }
        )
    return output


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
    return {"generated_at": datetime.now(UTC).isoformat(), "status": status, "thresholds": {"numeric_standardized_mean_shift": 0.2, "mean_prediction_shift": 0.03}, "missing_cells": {"validation": int(validation.isna().sum().sum()), "test": int(test.isna().sum().sum())}, "mean_prediction": {"validation": float(validation.predicted_make_probability.mean()), "test": float(test.predicted_make_probability.mean()), "absolute_shift": prediction_shift}, "feature_drift": drift}


def build_player_hex_bins(test: pd.DataFrame, player_ids: set[int]) -> pd.DataFrame:
    shots = test[test.player_id.isin(player_ids)].copy()
    shots["court_x"] = shots.loc_x / 10
    shots["court_y"] = shots.loc_y / 10
    shots = shots[shots.court_x.between(-25, 25) & shots.court_y.between(-4.75, 42.25)]
    shots["x_index"] = (shots.court_x / 2.5).round().astype(int)
    shots["x_bin"] = shots.x_index * 2.5
    shots["y_bin"] = ((shots.court_y - shots.x_index.mod(2) * 1.25) / 2.5).round() * 2.5 + shots.x_index.mod(2) * 1.25
    bins = empirical_bayes(shots, ["player_id", "x_bin", "y_bin"])
    return bins[bins.attempts >= 3][["player_id", "x_bin", "y_bin", "attempts", "actual_makes", "expected_makes", "extra_makes_per_100"]]


def export_app_bundle(prediction_path: Path = PROCESSED_DIR / "model_predictions.parquet", app_dir: Path = APP_DATA_DIR, metrics_path: Path = ARTIFACTS_DIR / "metrics.json") -> tuple[pd.DataFrame, dict]:
    scored = pd.read_parquet(prediction_path)
    briefs, areas = build_player_briefs(scored)
    test = scored[scored.split == "test"]
    validation = scored[scored.split == "validation"]
    hex_bins = build_player_hex_bins(test, set(briefs.player_id))
    monitoring = monitoring_report(scored)
    standards = {
        "label_rule": "Strong positive when the entire 80% adjusted interval is above zero; strong negative when entirely below zero; otherwise inconclusive.",
        "interval_level": "80%",
        "minimum_attempts": MIN_TEST_ATTEMPTS,
        "threshold_analysis": threshold_analysis(validation),
        "interval_sensitivity": interval_sensitivity(test),
        "profile_rule": "Dominant shot-area frequency at or above the validation-season 75th percentile among players at the same position; otherwise balanced.",
    }
    metrics = json.loads(metrics_path.read_text())
    app_dir.mkdir(parents=True, exist_ok=True)
    briefs.to_parquet(app_dir / "player_briefs.parquet", index=False)
    areas.to_parquet(app_dir / "player_areas.parquet", index=False)
    hex_bins.to_parquet(app_dir / "player_hex_bins.parquet", index=False)
    (app_dir / "validation_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (app_dir / "monitoring.json").write_text(json.dumps(monitoring, indent=2) + "\n")
    (app_dir / "evidence_standards.json").write_text(json.dumps(standards, indent=2) + "\n")
    metadata = {"model_version": metrics["model_version"], "created_at": metrics["created_at"], "qualified_players": int(len(briefs)), "minimum_test_attempts": MIN_TEST_ATTEMPTS, "evaluation_season": TEST_SEASON, "validation_season": metrics["seasons"]["validation"], "runtime_network_calls": False, "public_estimate_seasons": [TEST_SEASON]}
    (app_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return briefs, monitoring
