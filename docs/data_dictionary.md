# Data dictionary

## Raw shot identity

- `season`, `season_type`: competition period
- `game_id`, `game_event_id`, `shot_id`: stable shot identifiers
- `game_date`: chronological split field
- `player_id`, `player_name`, `team_id`, `team_abbreviation`: descriptive identifiers excluded from model features

## Model inputs

- `shot_distance`, `loc_x`, `loc_y`, `shot_angle`: spatial context
- `shot_value`, `is_corner_three`: scoring and geometry context
- `period`, `seconds_remaining_period`, `is_late_period`: game-time context
- `is_home`: location context
- `shot_zone_basic`, `shot_zone_area`, `shot_zone_range`: NBA shot zones
- `action_group`: normalized public action label
- `shot_family`: rim, midrange, corner three, or above the break

## Player evidence

- `actual_makes`, `expected_makes`: observed makes and summed calibrated probabilities
- `raw_extra_makes`: actual minus expected makes
- `adjusted_extra_makes`: empirical-Bayes-adjusted total difference
- `extra_makes_per_100`: adjusted difference per 100 attempts
- `lower_80`, `upper_80`: 80% adjusted interval
- `probability_positive`: posterior probability the adjusted rate is above zero
- `evidence_label`: strong positive, inconclusive, or strong negative under the frozen interval rule
- `attempts`: untouched 2025–26 field-goal attempts

## Shot distribution and area evidence

- `rim_share`, `midrange_share`, `three_share`: share of player attempts
- `*_percentile`: frequency percentile against validation-season players at the same position
- `shot_profile`: validation-benchmarked rim-heavy, midrange-heavy, perimeter-heavy, or balanced label
- `shot_area`: rim, midrange, left corner, right corner, or above the break
- `shot_share`: area attempts divided by player attempts
- `position_frequency_percentile`: area frequency percentile among the same position
- `is_top_two_volume`: whether the area is one of the player's two highest-volume areas

## Display-only roster fields

- `team_name`: latest team observed in the 2025–26 regular season
- `position`: public 2025–26 roster label normalized to Guard, Forward, or Center
