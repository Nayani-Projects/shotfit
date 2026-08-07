"""Fetch compact public roster data used only during bundle creation."""

from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

from shotfit.config import REFERENCE_DIR, TEST_SEASON

POSITION_LABELS = {"G": "Guard", "F": "Forward", "C": "Center"}
# Orlando Robinson appeared for Toronto but was absent from season-ending rosters.
POSITION_OVERRIDES = {
    1631115: "Center",
    1631093: "Guard",
    1630560: "Guard",
    1631246: "Guard",
}


def fetch_player_positions(season: str = TEST_SEASON) -> pd.DataFrame:
    """Cache one G/F/C position per player from season-ending team rosters."""
    rows: list[pd.DataFrame] = []
    for team in teams.get_teams():
        roster = commonteamroster.CommonTeamRoster(
            team_id=team["id"], season=season, timeout=60
        ).get_data_frames()[0]
        rows.append(
            roster[["PLAYER_ID", "PLAYER", "POSITION"]].assign(
                roster_team_id=team["id"], roster_team=team["full_name"]
            )
        )
        time.sleep(0.6)
    positions = pd.concat(rows, ignore_index=True).rename(
        columns={"PLAYER_ID": "player_id", "PLAYER": "roster_player_name", "POSITION": "position"}
    )
    positions["position"] = positions.position.str[0].map(POSITION_LABELS)
    positions = pd.concat(
        [
            positions,
            pd.DataFrame(
                [
                    {"player_id": player_id, "position": position}
                    for player_id, position in POSITION_OVERRIDES.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    positions = positions.drop_duplicates("player_id", keep="last")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(REFERENCE_DIR / f"player_positions_{season.replace('-', '_')}.parquet", index=False)
    return positions
