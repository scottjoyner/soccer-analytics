from pathlib import Path
from typing import Any

import pandas as pd


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


def series_from_candidates(frame: pd.DataFrame, candidates: list[str], nested_paths: dict[str, list[str]] | None = None) -> pd.Series:
    nested_paths = nested_paths or {}
    for column in candidates:
        if column in frame.columns:
            return frame[column].map(value_name)
        base = column.split(".")[0]
        if base in frame.columns and column in nested_paths:
            return frame[base].map(lambda value: nested_value(value, nested_paths[column]))
    return pd.Series([None] * len(frame), index=frame.index)


def normalize_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    normalized = events.copy()
    normalized["_match_id"] = series_from_candidates(normalized, ["match_id"])
    normalized["_player_name"] = series_from_candidates(normalized, ["player.name", "player_name", "player"], {"player.name": ["name"]})
    normalized["_team_name"] = series_from_candidates(normalized, ["team.name", "team_name", "team"], {"team.name": ["name"]})
    normalized["_event_type"] = series_from_candidates(normalized, ["type.name", "event_type", "type"], {"type.name": ["name"]})
    normalized["_shot_outcome"] = series_from_candidates(normalized, ["shot.outcome.name", "shot_outcome"], {"shot.outcome.name": ["outcome", "name"]})
    normalized["_pass_outcome"] = series_from_candidates(normalized, ["pass.outcome.name", "pass_outcome"], {"pass.outcome.name": ["outcome", "name"]})
    if "minute" in normalized.columns:
        normalized["_minute"] = pd.to_numeric(normalized["minute"], errors="coerce")
    else:
        normalized["_minute"] = 0.0
    return normalized


def build_player_match_stats(events: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_event_frame(events)
    normalized = normalized.dropna(subset=["_match_id", "_player_name"])
    if normalized.empty:
        return pd.DataFrame(
            columns=[
                "match_id",
                "player_name",
                "team_name",
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
                "max_minute",
            ]
        )
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
    grouped = normalized.groupby(["_match_id", "_player_name", "_team_name"], dropna=False)
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
    ).reset_index()
    output = output.rename(columns={"_match_id": "match_id", "_player_name": "player_name", "_team_name": "team_name"})
    output["pass_completion_rate"] = output["completed_passes"] / output["passes"].replace(0, pd.NA)
    output["pass_completion_rate"] = output["pass_completion_rate"].fillna(0.0)
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


def write_player_match_stats(events_path: Path, output_path: Path) -> Path:
    events = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    stats = build_player_match_stats(events)
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
