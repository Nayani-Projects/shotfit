# ShotFit model card

## Intended use

ShotFit estimates whether a player's public shooting results exceeded what an average NBA shooter would be expected to make from the same observable shot profile. It is designed to focus film, tracking, and scouting review.

It is not a causal projection, acquisition grade, or substitute for team tracking, medical, coaching, or scouting data.

## Data and validation

- Source: NBA.com public statistics accessed through `nba_api`
- Train: 2022–23 regular season
- Validation and calibration: 2023–24 regular season
- Untouched test: 2024–25 regular season
- Unit: one field-goal attempt
- Target: made or missed

Player and team identity are excluded. Features describe location, distance, shot family, action label, period, time remaining, and home/road context.

## Model selection

The pipeline compares a smoothed zone-average baseline, regularized logistic regression, and XGBoost. XGBoost is selected only when it materially improves both validation log loss and Brier score; otherwise the simpler logistic model is retained. The selected model is calibrated with isotonic regression before the untouched test.

## Player estimates

Expected makes are summed model probabilities. Actual-minus-expected rates are stabilized with a normal-normal empirical-Bayes model, producing adjusted extra makes per 100 attempts and 80% intervals.

## Limitations

Public shot records omit shot-level defender distance, pass quality, movement, player balance, screen quality, exact play design, fatigue, health, and internal role information. Action labels are coarse. The automatic role is a transparent descriptive prompt, not an optimized or causal recommendation.

