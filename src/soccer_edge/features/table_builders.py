import pandas as pd

from soccer_edge.player_stats import aggregate_roster_to_team


def build_prematch_table(
    matches: pd.DataFrame,
    team_features: pd.DataFrame | None = None,
    player_features: pd.DataFrame | None = None,
    player_team_col: str = "team_name",
) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    required = {"match_id", "home_team", "away_team"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"missing match columns: {sorted(missing)}")

    output = matches.copy()
    if team_features is not None and not team_features.empty and "team" in team_features.columns:
        home_features = team_features.add_prefix("home_").rename(columns={"home_team": "home_team"})
        away_features = team_features.add_prefix("away_").rename(columns={"away_team": "away_team"})
        output = output.merge(home_features, on="home_team", how="left")
        output = output.merge(away_features, on="away_team", how="left")

    if player_features is not None and not player_features.empty and player_team_col in player_features.columns:
        roster = aggregate_roster_to_team(player_features, team_col=player_team_col)
        feature_columns = [column for column in roster.columns if column != player_team_col]
        home_roster = roster.rename(columns={player_team_col: "home_team"})[["home_team", *feature_columns]].copy()
        home_roster.columns = ["home_team"] + [f"home_{column}" for column in feature_columns]
        away_roster = home_roster.rename(columns=lambda column: column.replace("home_", "away_", 1))
        output = output.merge(home_roster, on="home_team", how="left")
        output = output.merge(away_roster, on="away_team", how="left")
    return output


def build_inplay_rolling_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_seconds: float,
    group_column: str = "match_id",
    time_column: str = "timestamp_seconds",
    carry_columns: list[str] | None = None,
) -> pd.DataFrame:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    carry_columns = [] if carry_columns is None else carry_columns
    required = {group_column, time_column, *feature_columns, *carry_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    ordered = frame.sort_values([group_column, time_column]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        group_value = row[group_column]
        current_time = float(row[time_column])
        start_time = current_time - window_seconds
        window = ordered[
            (ordered[group_column] == group_value)
            & (ordered[time_column] <= current_time)
            & (ordered[time_column] > start_time)
        ]
        output_row: dict[str, object] = {group_column: group_value, time_column: current_time}
        for column in carry_columns:
            output_row[column] = row[column]
        for column in feature_columns:
            output_row[f"{column}_mean_{int(window_seconds)}s"] = float(window[column].mean())
            output_row[f"{column}_last"] = float(row[column])
        rows.append(output_row)
    return pd.DataFrame(rows)
