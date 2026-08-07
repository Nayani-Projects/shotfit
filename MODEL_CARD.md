# ShotFit model card

## Intended use

ShotFit identifies players whose 2025–26 made-shot totals provide positive, negative, or inconclusive evidence relative to calibrated expectation from the same observable public shot context. It is intended to focus descriptive film and tracking review.

It is not a forecast, causal projection, role recommendation, acquisition grade, or substitute for team tracking, medical, coaching, and scouting information.

## Data and chronological split

- Source: NBA.com public statistics through `nba_api`
- Training: 2023–24 regular season
- Validation, calibration, and standards: 2024–25 regular season
- Untouched public evidence: 2025–26 regular season
- Unit: field-goal attempt
- Target: made or missed

Player and team identity are excluded. Model inputs describe location, distance, shot value and family, normalized public action label, period, time remaining, and home/road context.

## Model selection and evaluation

The pipeline compares a zone-average baseline, regularized logistic regression, and XGBoost. XGBoost must materially improve validation log loss and Brier score or the simpler logistic model is retained. The selected model is calibrated with isotonic regression before untouched testing.

## Player and area evidence

Expected makes are summed calibrated probabilities. Actual-minus-expected rates are stabilized with a normal-normal empirical-Bayes model. The public bundle includes actual, expected, raw difference, adjusted difference, adjusted difference per 100, probability of being positive, and an 80% adjusted interval.

Evidence labels are mechanical:

- **Strong positive evidence:** lower 80% bound is above zero.
- **Strong negative evidence:** upper 80% bound is below zero.
- **Inconclusive evidence:** interval crosses zero.

The 80% level is used because this is a screening product, not a final decision. The app reports 90% and 95% classification sensitivity.

## Eligibility and shot profiles

The 250-attempt floor was selected on validation data before viewing final player labels. The technical tab reports eligible-player count, median interval width, conclusive share, and deterministic split-half correlation at 150, 250, 400, and 600 attempts.

Shot profiles are descriptive. A player is rim-, midrange-, or perimeter-heavy only when the corresponding frequency reaches the validation-season 75th percentile among the same position; otherwise the profile is balanced.

## Limitations

Public records omit shot-level defender distance, pass quality, movement, balance, screen quality, exact play design, fatigue, health, and internal role information. Action labels are coarse. The normal-normal adjustment is an approximation, and the 80% screening level intentionally accepts more uncertainty than a 95% scientific interval. Results describe 2025–26 evidence only.
