from pathlib import Path
from typing import Any

import pandas as pd

COUNT_METRICS = [
    "total_events",
    "shots",
    "goals",
    "passes",
    "completed_passes",
    "carries",
    "dribbles",
    "pressures",
    "interceptions",
    "tackles",
    "fouls_committed",
]


def normalize_per_90(value, minutes_played, default: float = 0.0):
    value_s = pd.Series(value) if pd.api.types.is_list_like(value) else pd.Series([value])
    minutes_s = pd.Series(minutes_played) if pd.api.types.is_list_like(minutes_played) else pd.Series([minutes_played])
    minutes_safe = minutes_s.where(minutes_s > 0, 1.0)
    result = value_s / minutes_safe * 90.0
    result = result.where(minutes_s > 0, 0.0)
    result = result.where(value_s.notna() & minutes_s.notna(), default)
    if not (pd.api.types.is_list_like(value) or pd.api.types.is_list_like(minutes_played)):
        return float(result.iloc[0])
    return result


def _coerce_minutes(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        minute, second = text.split(":", 1)
        try:
            return float(minute) + float(second) / 60.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def expected_starter_flag(lineup_df, player_id, player_column: str = "player_id", default: float = 0.5):
    if lineup_df is None or getattr(lineup_df, "empty", True):
        return default
    features = normalize_lineup_players(lineup_df, player_column=player_column, default_starter=default)
    if player_column not in features.columns or "is_expected_starter" not in features.columns:
        return default
    matched = features[features[player_column] == player_id]
    if matched.empty:
        return default
    return float(matched.iloc[0]["is_expected_starter"])


def _starter_flags(player_ids: pd.Series, lineup_df, lineup_player_column: str, default: float = 0.5) -> pd.Series:
    if lineup_df is None or getattr(lineup_df, "empty", True):
        return pd.Series([default] * len(player_ids), index=player_ids.index, dtype=float)
    features = normalize_lineup_players(lineup_df, player_column=lineup_player_column, default_starter=default)
    if lineup_player_column not in features.columns or "is_expected_starter" not in features.columns:
        return pd.Series([default] * len(player_ids), index=player_ids.index, dtype=float)
    mapping = features.dropna(subset=[lineup_player_column]).drop_duplicates(lineup_player_column).set_index(lineup_player_column)["is_expected_starter"]
    return player_ids.map(mapping).fillna(default).astype(float)


def _lineup_minutes(row: dict[str, object]) -> float | None:
    positions = row.get("positions")
    if not isinstance(positions, list):
        return None
    total = 0.0
    seen_interval = False
    for position in positions:
        if not isinstance(position, dict):
            continue
        start = _coerce_minutes(position.get("from"))
        end = _coerce_minutes(position.get("to"))
        if start is None:
            continue
        if end is None:
            end = 90.0
        total += max(0.0, end - start)
        seen_interval = True
    return total if seen_interval else None


def _lineup_starter(row: dict[str, object], default: float = 0.5) -> float:
    for key in ("is_expected_starter", "starter", "is_starter"):
        if key in row and pd.notna(row[key]):
            return float(bool(row[key])) if isinstance(row[key], bool) else float(row[key])
    positions = row.get("positions")
    if isinstance(positions, list) and positions:
        for position in positions:
            if not isinstance(position, dict):
                continue
            start = _coerce_minutes(position.get("from"))
            reason = str(position.get("start_reason", "")).lower()
            if start == 0.0 or "starting" in reason:
                return 1.0
        return 0.0
    return default


def normalize_lineup_players(
    lineup_df: pd.DataFrame | None,
    player_column: str = "player_id",
    default_starter: float = 0.5,
) -> pd.DataFrame:
    if lineup_df is None or getattr(lineup_df, "empty", True):
        return pd.DataFrame(columns=[player_column, "player_name", "team_name", "is_expected_starter", "minutes_played"])

    rows: list[dict[str, object]] = []
    if "lineup" in lineup_df.columns:
        for _, team_row in lineup_df.iterrows():
            team_name = team_row.get("team_name") or team_row.get("team") or ""
            if isinstance(team_name, dict):
                team_name = team_name.get("name") or team_name.get("id") or ""
            lineup = team_row.get("lineup")
            if not isinstance(lineup, list):
                continue
            for player in lineup:
                if not isinstance(player, dict):
                    continue
                row = {
                    "player_id": player.get("player_id") or player.get("id"),
                    "player_name": player.get("player_name") or player.get("name"),
                    "team_name": team_name,
                    "positions": player.get("positions"),
                }
                row["is_expected_starter"] = _lineup_starter(row, default=default_starter)
                row["minutes_played"] = _lineup_minutes(row)
                rows.append(row)
    else:
        for _, raw_row in lineup_df.iterrows():
            row = raw_row.to_dict()
            normalized = {
                "player_id": row.get("player_id") or row.get("id") or row.get(player_column),
                "player_name": row.get("player_name") or row.get("player") or row.get("name"),
                "team_name": row.get("team_name") or row.get("team") or "",
                "is_expected_starter": _lineup_starter(row, default=default_starter),
                "minutes_played": row.get("minutes_played") or row.get("mins_played") or row.get("duration_minutes"),
            }
            if isinstance(normalized["team_name"], dict):
                normalized["team_name"] = normalized["team_name"].get("name") or normalized["team_name"].get("id") or ""
            rows.append(normalized)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[player_column, "player_name", "team_name", "is_expected_starter", "minutes_played"])
    frame["minutes_played"] = pd.to_numeric(frame.get("minutes_played"), errors="coerce").fillna(0.0)
    if player_column != "player_id" and player_column not in frame.columns:
        frame[player_column] = frame.get("player_id") if player_column == "player_id" else frame.get("player_name")
    return frame


def aggregate_roster_to_team(
    player_features_df: pd.DataFrame,
    team_col: str = "team_name",
    starter_col: str = "is_expected_starter",
) -> pd.DataFrame:
    if team_col not in player_features_df.columns:
        raise ValueError(f"missing team column: {team_col}")
    numeric_columns = [column for column in player_features_df.select_dtypes(include="number").columns if column != team_col]
    grouped = player_features_df.groupby(team_col)[numeric_columns]
    agg = grouped.agg(["mean", "min", "max"]).reset_index()
    flattened: list[str] = [team_col]
    for column in numeric_columns:
        for stat in ("mean", "min", "max"):
            flattened.append(f"{column}_{stat}")
    agg.columns = flattened

    if starter_col in player_features_df.columns:
        counts = (
            player_features_df[player_features_df[starter_col] >= 0.5]
            .groupby(team_col)[starter_col]
            .size()
            .reset_index(name="expected_starter_count")
        )
        agg = agg.merge(counts, on=team_col, how="left")
        agg["expected_starter_count"] = agg["expected_starter_count"].fillna(0).astype(int)
    return agg


def nested_value(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def value_name(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("name") or value.get("id")
    return value


def value_id(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("id") or value.get("player_id") or value.get("name")
    return value


def series_from_candidates(frame: pd.DataFrame, candidates: list[str], nested_paths: dict[str, list[str]] | None = None) -> pd.Series:
    nested_paths = nested_paths or {}
    for column in candidates:
        if column in frame.columns:
            return frame[column].map(value_name)
        base = column.split(".")[0]
        if base in frame.columns and column in nested_paths:
            return frame[base].map(lambda value: nested_value(value, nested_paths[column]))
    return pd.Series([None] * len(frame), index=frame.index)


def id_series_from_candidates(frame: pd.DataFrame, candidates: list[str], nested_paths: dict[str, list[str]] | None = None) -> pd.Series:
    nested_paths = nested_paths or {}
    for column in candidates:
        if column in frame.columns:
            return frame[column].map(value_id)
        base = column.split(".")[0]
        if base in frame.columns and column in nested_paths:
            return frame[base].map(lambda value: nested_value(value, nested_paths[column]))
    return pd.Series([None] * len(frame), index=frame.index)


def normalize_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    normalized = events.copy()
    normalized["_match_id"] = series_from_candidates(normalized, ["match_id"])
    normalized["_player_id"] = id_series_from_candidates(normalized, ["player.id", "player_id"], {"player.id": ["id"]})
    normalized["_player_name"] = series_from_candidates(normalized, ["player.name", "player_name", "player"], {"player.name": ["name"]})
    normalized["_team_name"] = series_from_candidates(normalized, ["team.name", "team_name", "team"], {"team.name": ["name"]})
    normalized["_event_type"] = series_from_candidates(normalized, ["type.name", "event_type", "type"], {"type.name": ["name"]})
    normalized["_shot_outcome"] = series_from_candidates(normalized, ["shot.outcome.name", "shot_outcome"], {"shot.outcome.name": ["outcome", "name"]})
    normalized["_pass_outcome"] = series_from_candidates(normalized, ["pass.outcome.name", "pass_outcome"], {"pass.outcome.name": ["outcome", "name"]})
    if "minute" in normalized.columns:
        normalized["_minute"] = pd.to_numeric(normalized["minute"], errors="coerce")
    else:
        normalized["_minute"] = 0.0
    minute_col = next((column for column in ["minutes_played", "mins_played", "duration_minutes"] if column in normalized.columns), None)
    normalized["_minutes_played"] = pd.to_numeric(normalized[minute_col], errors="coerce") if minute_col else pd.NA
    return normalized


def _empty_player_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "player_id",
            "player_name",
            "team_name",
            *COUNT_METRICS,
            "max_minute",
            "minutes_played",
            "pass_completion_rate",
            "is_expected_starter",
        ]
    )


def _merge_lineup_features(output: pd.DataFrame, lineup: pd.DataFrame | None, lineup_player_column: str) -> pd.DataFrame:
    output = output.copy()
    output["is_expected_starter"] = 0.5
    output["minutes_played"] = output["minutes_played"].fillna(0.0)
    lineup_features = normalize_lineup_players(lineup, player_column=lineup_player_column)
    if lineup_features.empty:
        return output

    if "player_id" in output.columns and "player_id" in lineup_features.columns and output["player_id"].notna().any():
        merge_key = "player_id"
    else:
        merge_key = "player_name"
    if merge_key not in lineup_features.columns:
        return output

    merge_columns = [merge_key, "is_expected_starter", "minutes_played"]
    lineup_subset = lineup_features[merge_columns].drop_duplicates(merge_key)
    merged = output.merge(lineup_subset, on=merge_key, how="left", suffixes=("", "_lineup"))
    merged["is_expected_starter"] = merged["is_expected_starter_lineup"].combine_first(merged["is_expected_starter"])
    merged["minutes_played"] = merged["minutes_played_lineup"].where(merged["minutes_played_lineup"] > 0, merged["minutes_played"])
    return merged.drop(columns=[column for column in ["is_expected_starter_lineup", "minutes_played_lineup"] if column in merged.columns])


def build_player_match_stats(
    events: pd.DataFrame,
    lineup: pd.DataFrame | None = None,
    lineup_player_column: str = "player_id",
    player_id_column: str = "player_id",
) -> pd.DataFrame:
    normalized = normalize_event_frame(events)
    normalized = normalized.dropna(subset=["_match_id", "_player_name"])
    if normalized.empty:
        return _empty_player_stats()
    normalized["_is_shot"] = normalized["_event_type"].eq("Shot")
    normalized["_is_goal"] = normalized["_is_shot"] & normalized["_shot_outcome"].eq("Goal")
    normalized["_is_pass"] = normalized["_event_type"].eq("Pass")
    normalized["_is_completed_pass"] = normalized["_is_pass"] & normalized["_pass_outcome"].isna()
    normalized["_is_carry"] = normalized["_event_type"].eq("Carry")
    normalized["_is_dribble"] = normalized["_event_type"].eq("Dribble")
    normalized["_is_pressure"] = normalized["_event_type"].eq("Pressure")
    normalized["_is_interception"] = normalized["_event_type"].eq("Interception")
    normalized["_is_tackle"] = normalized["_event_type"].eq("Duel")
    normalized["_is_foul"] = normalized["_event_type"].eq("Foul Committed")
    grouped = normalized.groupby(["_match_id", "_player_id", "_player_name", "_team_name"], dropna=False)
    output = grouped.agg(
        total_events=("_event_type", "size"),
        shots=("_is_shot", "sum"),
        goals=("_is_goal", "sum"),
        passes=("_is_pass", "sum"),
        completed_passes=("_is_completed_pass", "sum"),
        carries=("_is_carry", "sum"),
        dribbles=("_is_dribble", "sum"),
        pressures=("_is_pressure", "sum"),
        interceptions=("_is_interception", "sum"),
        tackles=("_is_tackle", "sum"),
        fouls_committed=("_is_foul", "sum"),
        max_minute=("_minute", "max"),
        minutes_played=("_minutes_played", "max"),
    ).reset_index()
    output = output.rename(
        columns={
            "_match_id": "match_id",
            "_player_id": "player_id",
            "_player_name": "player_name",
            "_team_name": "team_name",
        }
    )
    output["pass_completion_rate"] = output["completed_passes"] / output["passes"].replace(0, pd.NA)
    output["pass_completion_rate"] = output["pass_completion_rate"].fillna(0.0)
    output = _merge_lineup_features(output, lineup, lineup_player_column)
    output["minutes_played"] = pd.to_numeric(output["minutes_played"], errors="coerce").fillna(0.0)
    for column in COUNT_METRICS:
        if column in output.columns:
            output[f"{column}_per_90"] = normalize_per_90(output[column], output["minutes_played"])
            output[f"{column}_per_observed90"] = normalize_per_90(output[column], output["max_minute"])
    if player_id_column not in output.columns:
        output[player_id_column] = output["player_name"]
    return output


def build_player_form_features(
    player_match_stats: pd.DataFrame,
    window: int = 5,
    player_column: str = "player_name",
    order_column: str = "match_id",
) -> pd.DataFrame:
    if player_column not in player_match_stats.columns:
        raise ValueError(f"missing player column: {player_column}")
    if order_column not in player_match_stats.columns:
        raise ValueError(f"missing order column: {order_column}")
    frame = player_match_stats.sort_values([player_column, order_column]).copy()
    metric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {order_column}
    ]
    for column in metric_columns:
        frame[f"{column}_form_{window}"] = (
            frame.groupby(player_column)[column]
            .transform(lambda values: values.shift(1).rolling(window=window, min_periods=1).mean())
            .fillna(0.0)
        )
    return frame


def build_team_player_feature_table(
    player_stats: pd.DataFrame,
    match_column: str = "match_id",
    team_column: str = "team_name",
) -> pd.DataFrame:
    missing = [column for column in [match_column, team_column] if column not in player_stats.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    numeric_columns = [column for column in player_stats.select_dtypes(include="number").columns if column != match_column]
    grouped = player_stats.groupby([match_column, team_column], dropna=False)[numeric_columns].sum().reset_index()
    grouped["player_count"] = player_stats.groupby([match_column, team_column], dropna=False)["player_name"].nunique().values
    return grouped


def assign_match_opponents(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add an ``opponent_team`` column: for each match, the other team(s) a player
    faced (joined with ';' if more than one). Requires ``match_id`` and ``team_name``."""

    if "match_id" not in player_stats.columns or "team_name" not in player_stats.columns:
        raise ValueError("match_id and team_name are required to compute opponents")
    frame = player_stats.copy()
    match_teams = frame.groupby("match_id")["team_name"].apply(lambda values: sorted(set(values.dropna())))

    def opponent_for(row: pd.Series) -> str:
        teams = match_teams.get(row["match_id"], [])
        others = [team for team in teams if team != row["team_name"]]
        return ";".join(others)

    frame["opponent_team"] = frame.apply(opponent_for, axis=1)
    return frame


def build_player_aggregates(
    player_stats: pd.DataFrame,
    group_keys: list[str] | None = None,
    split_by: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-match player stats into cross-match summaries.

    For each count metric this produces ``total_<metric>`` (career/season sum) and
    ``avg_<metric>`` (per-match average), plus ``appearances`` (distinct matches) and
    an overall ``pass_completion_rate``. Source-agnostic: feed it the output of
    ``build_player_match_stats`` from any open event feed (StatsBomb, Metrica, ...).

    Use ``split_by=["opponent"]`` to break totals out per opponent faced, or
    ``split_by=["team"]`` to separate stints by team (useful for transfers).
    """

    split_by_set = {value.strip() for value in (split_by or []) if value.strip()}
    group_keys = list(group_keys or ["player_name"])
    if "player_name" not in player_stats.columns:
        raise ValueError("player_stats must include a player_name column")

    frame = player_stats.copy()
    if "opponent" in split_by_set:
        if "match_id" not in frame.columns or "team_name" not in frame.columns:
            raise ValueError("match_id and team_name are required to split by opponent")
        frame = assign_match_opponents(frame)
        if "opponent_team" not in group_keys:
            group_keys.append("opponent_team")
    if "team" in split_by_set and "team_name" not in group_keys:
        group_keys.append("team_name")

    missing = [key for key in group_keys if key not in frame.columns]
    if missing:
        raise ValueError(f"missing group columns: {missing}")

    count_metrics = [column for column in COUNT_METRICS if column in frame.columns]

    grouped = frame.groupby(group_keys, dropna=False)
    order_column = "match_id" if "match_id" in frame.columns else "player_name"
    agg = grouped.agg(
        appearances=(order_column, "nunique" if order_column == "match_id" else "size"),
        **{f"total_{column}": (column, "sum") for column in count_metrics},
    ).reset_index()

    for column in count_metrics:
        agg[f"avg_{column}"] = agg[f"total_{column}"] / agg["appearances"].replace(0, pd.NA)

    if "minutes_played" in frame.columns:
        total_minutes = grouped["minutes_played"].sum().reset_index(name="total_minutes")
        agg = agg.merge(total_minutes, on=group_keys)
        for column in count_metrics:
            agg[f"{column}_per_90"] = normalize_per_90(agg[f"total_{column}"], agg["total_minutes"])
    elif "max_minute" in frame.columns:
        observed_minutes = grouped["max_minute"].sum().reset_index(name="observed_event_minutes")
        agg = agg.merge(observed_minutes, on=group_keys)
        for column in count_metrics:
            agg[f"{column}_per_observed90"] = normalize_per_90(agg[f"total_{column}"], agg["observed_event_minutes"])

    if "completed_passes" in frame.columns and "passes" in frame.columns:
        agg["pass_completion_rate"] = agg["total_completed_passes"] / agg["total_passes"].replace(0, pd.NA)

    if "team_name" in frame.columns:
        if "opponent_team" in group_keys:
            team = grouped["team_name"].apply(lambda values: ",".join(sorted(set(values.dropna())))).reset_index(name="team")
            agg = agg.merge(team, on=group_keys)
        elif "team_name" not in group_keys:
            teams = grouped["team_name"].apply(lambda values: ",".join(sorted(set(values.dropna())))).reset_index(name="teams")
            agg = agg.merge(teams, on=group_keys)

    numeric_columns = [column for column in agg.columns if column not in group_keys]
    agg[numeric_columns] = agg[numeric_columns].fillna(0.0)
    return agg


def write_player_match_stats(events_path: Path, output_path: Path, lineup_path: Path | None = None) -> Path:
    events = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    lineup = None
    if lineup_path is not None:
        lineup = pd.read_parquet(lineup_path) if lineup_path.suffix == ".parquet" else pd.read_csv(lineup_path)
    stats = build_player_match_stats(events, lineup=lineup)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        stats.to_parquet(output_path, index=False)
    else:
        stats.to_csv(output_path, index=False)
    return output_path


def write_player_form_features(player_stats_path: Path, output_path: Path, window: int = 5, order_column: str = "match_id") -> Path:
    stats = pd.read_parquet(player_stats_path) if player_stats_path.suffix == ".parquet" else pd.read_csv(player_stats_path)
    form = build_player_form_features(stats, window=window, order_column=order_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        form.to_parquet(output_path, index=False)
    else:
        form.to_csv(output_path, index=False)
    return output_path


def build_player_aggregates_from_events(event_frames: list[pd.DataFrame], group_keys: list[str] | None = None) -> pd.DataFrame:
    if not event_frames:
        raise ValueError("at least one event frame is required")
    per_match = [build_player_match_stats(frame) for frame in event_frames]
    combined = pd.concat(per_match, ignore_index=True)
    return build_player_aggregates(combined, group_keys=group_keys)


def write_player_aggregates(player_stats_path: Path, output_path: Path, group_keys: list[str] | None = None) -> Path:
    stats = pd.read_parquet(player_stats_path) if player_stats_path.suffix == ".parquet" else pd.read_csv(player_stats_path)
    aggregates = build_player_aggregates(stats, group_keys=group_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        aggregates.to_parquet(output_path, index=False)
    else:
        aggregates.to_csv(output_path, index=False)
    return output_path
