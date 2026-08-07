# Data dictionary

## Raw shot identity

- `season`, `season_type`: competition period
- `game_id`, `game_event_id`, `shot_id`: stable shot identifiers
- `game_date`: date used for chronological splitting
- `player_id`, `player_name`, `team_id`, `team_abbreviation`: descriptive identifiers excluded from model features

## Model inputs

- `shot_distance`, `loc_x`, `loc_y`, `shot_angle`: spatial context
- `shot_value`, `is_corner_three`: scoring and geometry context
- `period`, `seconds_remaining_period`, `is_late_period`: game-time context
- `is_home`: location context
- `shot_zone_basic`, `shot_zone_area`, `shot_zone_range`: NBA shot zones
- `action_group`: normalized public action label
- `shot_family`: at rim, midrange, corner three, or above the break

## Target and outputs

- `shot_made`: binary target
- `predicted_make_probability`: calibrated expected make probability
- `extra_makes_per_100`: reliability-adjusted player residual rate
- `lower_80`, `upper_80`: 80% posterior interval
- `role`: descriptive role to investigate
- `team_name`: player's latest team observed during the 2025–26 regular season; display/filter only
- `position`: primary Guard, Forward, or Center label from public 2025–26 team rosters; display/filter only
