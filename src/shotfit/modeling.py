"""Chronological model comparison, calibration, and player evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from shotfit.config import (
    ARTIFACTS_DIR,
    PROCESSED_DIR,
    TEST_SEASON,
    TRAIN_SEASON,
    VALIDATION_SEASON,
)
from shotfit.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

ZONE_KEYS = ["shot_zone_basic", "shot_zone_area", "shot_zone_range", "action_group"]


@dataclass
class ZoneBaseline:
    global_rate: float
    rates: dict[tuple[str, ...], tuple[float, int]]
    strength: float = 50.0

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> ZoneBaseline:
        global_rate = float(frame["shot_made"].mean())
        grouped = frame.groupby(ZONE_KEYS)["shot_made"].agg(["mean", "count"])
        rates = {tuple(map(str, index)): (float(row["mean"]), int(row["count"])) for index, row in grouped.iterrows()}
        return cls(global_rate, rates)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = []
        for key in frame[ZONE_KEYS].astype(str).itertuples(index=False, name=None):
            rate, count = self.rates.get(tuple(key), (self.global_rate, 0))
            values.append((rate * count + self.global_rate * self.strength) / (count + self.strength))
        return np.asarray(values)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, list(NUMERIC_FEATURES)), ("categorical", categorical, list(CATEGORICAL_FEATURES))])


def logistic_model() -> Pipeline:
    return Pipeline([("features", _preprocessor()), ("model", LogisticRegression(C=1.0, max_iter=500))])


def xgboost_model() -> Pipeline:
    return Pipeline([
        ("features", _preprocessor()),
        ("model", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, min_child_weight=8, reg_lambda=4.0, eval_metric="logloss", n_jobs=-1, random_state=42)),
    ])


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    cuts = np.linspace(0, 1, bins + 1)
    indexes = np.clip(np.digitize(probability, cuts) - 1, 0, bins - 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        mask = indexes == index
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def metrics(y: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y)
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(actual, probability)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "calibration_error": expected_calibration_error(actual, probability),
    }


def calibration_table(y: pd.Series, probability: np.ndarray, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame({"actual": y.to_numpy(), "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], bins, duplicates="drop")
    out = frame.groupby("bin", observed=True).agg(predicted=("probability", "mean"), observed=("actual", "mean"), shots=("actual", "size")).reset_index(drop=True)
    return out.round(6).to_dict(orient="records")


def _fit_candidate(name: str, train: pd.DataFrame, validation: pd.DataFrame):
    if name == "Zone average":
        model = ZoneBaseline.fit(train)
        probability = model.predict(validation)
    elif name == "Logistic regression":
        model = logistic_model().fit(train[list(MODEL_FEATURES)], train["shot_made"])
        probability = model.predict_proba(validation[list(MODEL_FEATURES)])[:, 1]
    else:
        model = xgboost_model().fit(train[list(MODEL_FEATURES)], train["shot_made"])
        probability = model.predict_proba(validation[list(MODEL_FEATURES)])[:, 1]
    return model, probability


def choose_model(results: dict[str, dict[str, float]]) -> str:
    logistic = results["Logistic regression"]
    boosted = results["XGBoost"]
    if boosted["log_loss"] <= logistic["log_loss"] - 0.005 and boosted["brier_score"] <= logistic["brier_score"] - 0.001:
        return "XGBoost"
    return "Logistic regression"


def train_and_score(
    feature_path: Path = PROCESSED_DIR / "shot_features.parquet",
    artifacts_dir: Path = ARTIFACTS_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, dict]:
    shots = pd.read_parquet(feature_path)
    train = shots[shots.season == TRAIN_SEASON].copy()
    validation = shots[shots.season == VALIDATION_SEASON].copy()
    test = shots[shots.season == TEST_SEASON].copy()
    if not (train.game_date.max() < validation.game_date.min() < test.game_date.min()):
        raise ValueError("Chronological split failed")
    candidate_models = {}
    candidate_predictions = {}
    validation_metrics = {}
    for name in ("Zone average", "Logistic regression", "XGBoost"):
        model, probability = _fit_candidate(name, train, validation)
        candidate_models[name] = model
        candidate_predictions[name] = probability
        validation_metrics[name] = metrics(validation["shot_made"], probability)
    selected = choose_model(validation_metrics)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(candidate_predictions[selected], validation["shot_made"])
    validation_probability = np.asarray(calibrator.predict(candidate_predictions[selected]))
    train_validation = pd.concat([train, validation], ignore_index=True)
    if selected == "Logistic regression":
        final_model = logistic_model().fit(train_validation[list(MODEL_FEATURES)], train_validation["shot_made"])
    else:
        final_model = xgboost_model().fit(train_validation[list(MODEL_FEATURES)], train_validation["shot_made"])
    raw_test_probability = final_model.predict_proba(test[list(MODEL_FEATURES)])[:, 1]
    test_probability = np.asarray(calibrator.predict(raw_test_probability))
    validation_scored = validation.copy()
    validation_scored["split"] = "validation"
    validation_scored["predicted_make_probability"] = validation_probability
    test_scored = test.copy()
    test_scored["split"] = "test"
    test_scored["predicted_make_probability"] = test_probability
    scored = pd.concat([validation_scored, test_scored], ignore_index=True)
    scored["expected_points"] = scored.predicted_make_probability * scored.shot_value
    scored["actual_points"] = scored.shot_made * scored.shot_value
    test_metrics = metrics(test["shot_made"], test_probability)
    subgroup = []
    for family, group in test_scored.groupby("shot_family"):
        result = metrics(group.shot_made, group.predicted_make_probability.to_numpy())
        subgroup.append({"group": family, "shots": int(len(group)), **result})
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "calibrator": calibrator, "features": MODEL_FEATURES}, artifacts_dir / "shot_model.joblib")
    scored.to_parquet(processed_dir / "model_predictions.parquet", index=False)
    report = {
        "model_version": "shotfit-2026-08-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "selected_model": selected,
        "selection_rule": "XGBoost must improve validation log loss by 0.005 and Brier score by 0.001; otherwise prefer logistic regression.",
        "features": list(MODEL_FEATURES),
        "excluded": ["player identity", "team identity", "future statistics", "post-shot events"],
        "seasons": {"train": TRAIN_SEASON, "validation": VALIDATION_SEASON, "test": TEST_SEASON},
        "row_counts": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "validation_models": validation_metrics,
        "test_metrics": test_metrics,
        "calibration": calibration_table(test.shot_made, test_probability),
        "subgroups": subgroup,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return scored, report
