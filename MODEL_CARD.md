# ShotFit model card

## Model summary

ShotFit estimates how many shots an average NBA player would make from the public context attached to each field-goal attempt. It compares those expected makes with the player's actual makes, adjusts the difference for sample size, and reports the result by player and court area.

The model supports a review workflow. It helps basketball staff decide which shooting results deserve film and tracking work. It does not forecast future performance or make a personnel decision.

| Item | Value |
|---|---|
| Model version | `shotfit-2026-08-v3` |
| Selected model | Regularized logistic regression with isotonic calibration |
| Training season | 2023-24 regular season |
| Validation season | 2024-25 regular season |
| Final review season | 2025-26 regular season |
| Minimum player volume | 250 field-goal attempts |
| Qualified players | 319 |

## Intended use

ShotFit is designed for three tasks:

1. Find players whose made-shot totals differed from expectation.
2. Locate the court areas that contributed to the result.
3. Prepare specific questions for film, tracking data, and scouting review.

The output is descriptive. It should be combined with basketball context that is not available in public shot records.

### Uses outside the model's scope

ShotFit should not be used as:

- A forecast of next-season shooting
- A player ranking or acquisition grade
- A recommendation about role or shot selection
- A measure of overall offensive value
- A substitute for tracking, medical, coaching, or scouting information

## Data

ShotFit uses NBA.com public regular-season shot records collected through `nba_api`.

| Split | Season | Shots | Use |
|---|---|---:|---|
| Training | 2023-24 | 218,700 | Fit model parameters |
| Validation | 2024-25 | 219,527 | Compare models, calibrate probabilities, and set standards |
| Test | 2025-26 | 219,160 | Final model evaluation and public player results |

The chronological split prevents later seasons from influencing earlier model decisions. Public player and court-area results come only from the untouched 2025-26 test season.

### Target

The target is whether a field-goal attempt was made or missed.

### Model inputs

- Shot coordinates, distance, and angle
- Two-point or three-point value
- Corner-three indicator
- Public shot zone and action labels
- Shot family
- Period and time remaining
- Late-period indicator
- Home or road status

### Deliberate exclusions

- Player identity
- Team identity
- Future statistics
- Post-shot events

Excluding player and team identity makes the expected value a shot-context benchmark. It does not bake the shooter's historical reputation into the comparison.

## Model selection

The pipeline compares three candidates:

1. Zone-average baseline
2. Regularized logistic regression
3. XGBoost

The selection rule was set before final testing. XGBoost had to improve validation log loss by at least 0.005 and Brier score by at least 0.001. If it failed either requirement, the simpler logistic model would be retained.

| Validation model | Log loss | Brier score | ROC AUC | Calibration error |
|---|---:|---:|---:|---:|
| Zone average | 0.6461 | 0.2280 | 0.6461 | 0.0039 |
| Logistic regression | 0.6454 | 0.2278 | 0.6481 | 0.0104 |
| XGBoost | 0.6421 | 0.2264 | 0.6503 | 0.0096 |

XGBoost improved Brier score enough but improved log loss by only 0.0034 relative to logistic regression. It did not meet both requirements, so logistic regression was selected and calibrated with isotonic regression.

## Final evaluation

The selected model produced the following results on the untouched 2025-26 season:

| Metric | Result |
|---|---:|
| Log loss | 0.6441 |
| Brier score | 0.2270 |
| ROC AUC | 0.6521 |
| Calibration error | 0.0029 |

Calibration matters because expected makes are calculated by summing shot-level probabilities. A model can rank shots reasonably well and still produce misleading player totals if its probabilities are poorly calibrated.

## Player estimates

Expected makes are the sum of the selected model's probabilities. The unadjusted player difference is:

```text
actual makes - expected makes
```

ShotFit stabilizes player and court-area differences with a normal-normal empirical Bayes model. Smaller samples move farther toward zero and receive wider intervals. Larger samples retain more of the observed difference.

The app reports the adjusted difference as makes per 100 attempts because that scale is easier to compare across players.

## Review labels

The statistical rule uses an 80% adjusted interval:

- **Strong positive:** the full interval is above zero.
- **Strong negative:** the full interval is below zero.
- **Inconclusive:** the interval crosses zero.

The basketball-facing app translates those labels into **Worth reviewing**, **Potential concern**, and **No clear signal**.

The 80% level is used for screening rather than final decision-making. More conservative sensitivity checks are retained:

| Interval | Strong positive | Inconclusive | Strong negative |
|---|---:|---:|---:|
| 80% | 71 | 185 | 63 |
| 90% | 44 | 235 | 40 |
| 95% | 33 | 261 | 25 |

As the interval becomes more conservative, more players move into the inconclusive group.

## Minimum attempt standard

The minimum was selected with 2024-25 validation data before the 2025-26 labels were created.

| Minimum attempts | Eligible players | Median interval width | Conclusive share | Split-half correlation |
|---|---:|---:|---:|---:|
| 150 | 363 | 4.56 | 36.6% | 0.56 |
| **250** | **306** | **4.39** | **40.8%** | **0.62** |
| 400 | 232 | 4.17 | 42.2% | 0.64 |
| 600 | 130 | 3.71 | 50.8% | 0.71 |

The 250-attempt cutoff improved stability over 150 attempts while retaining substantially more players than the 400 and 600 attempt alternatives.

## Shot profiles

Shot profiles describe frequency, not performance. Rim, midrange, and three-point shares are compared with players at the same position in the validation season.

A player is labeled rim-heavy, midrange-heavy, or perimeter-heavy when that frequency reaches the 75th percentile for his position. Every other player is labeled balanced.

## Limitations

Public shot records leave out important context:

- Defender distance and contest quality
- Pass quality and advantage creation
- Movement, balance, and footwork
- Screen quality and play design
- Fatigue, injury, and health
- Coaching instructions and internal role expectations

Public action labels are also coarse, and the sample-size adjustment is an approximation. The 80% screening level accepts more uncertainty than a conventional 95% scientific interval.

Results describe 2025-26 only. They do not establish future shooting ability, causal team fit, or the shots a player should take.

## Runtime and reproducibility

The Streamlit app loads precomputed Parquet and JSON files. It does not query NBA.com or retrain the model at runtime.

The repository includes commands for ingestion, feature construction, training, evaluation, and bundle export. Unit tests cover feature logic, ingestion checks, interval labels, court output, and artifact reconciliation.
