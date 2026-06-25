import pandas as pd


def build_prematch_table(matches: pd.DataFrame, team_features: pd.DataFrame | None = None) -> pd.DataFrame:
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
    return output


def build_inplay_rolling_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_seconds: float,
    group_column: str = "match_id",
    time_column: str = "timestamp_seconds",
) -> pd.DataFrame:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    required = {group_column, time_column, *feature_columns}
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
        for column in feature_columns:
            output_row[f"{column}_mean_{int(window_seconds)}s"] = float(window[column].mean())
            output_row[f"{column}_last"] = float(row[column])
        rows.append(output_row)
    return pd.DataFrame(rows)
