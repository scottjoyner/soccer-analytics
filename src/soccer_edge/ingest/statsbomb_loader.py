import json
from pathlib import Path

import pandas as pd


def read_json_file(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_competitions(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "competitions.json"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(read_json_file(path))


def load_match_files(source_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    matches_dir = source_dir / "matches"
    if not matches_dir.exists():
        return pd.DataFrame()

    for path in sorted(matches_dir.glob("*/*.json")):
        data = read_json_file(path)
        if isinstance(data, list):
            rows.extend(data)
    return pd.DataFrame(rows)


def load_event_files(source_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    events_dir = source_dir / "events"
    if not events_dir.exists():
        return pd.DataFrame()

    for path in sorted(events_dir.glob("*.json")):
        match_id = path.stem
        data = read_json_file(path)
        if not isinstance(data, list):
            continue
        for event in data:
            event = dict(event)
            event["match_id"] = match_id
            rows.append(event)
    return pd.DataFrame(rows)


def load_lineup_files(source_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lineups_dir = source_dir / "lineups"
    if not lineups_dir.exists():
        return pd.DataFrame()

    for path in sorted(lineups_dir.glob("*.json")):
        match_id = path.stem
        data = read_json_file(path)
        if not isinstance(data, list):
            continue
        for team in data:
            team_row = dict(team)
            team_row["match_id"] = match_id
            rows.append(team_row)
    return pd.DataFrame(rows)


def ingest_statsbomb(source_dir: Path) -> dict[str, str]:
    competitions = load_competitions(source_dir)
    matches = load_match_files(source_dir)
    events = load_event_files(source_dir)
    lineups = load_lineup_files(source_dir)

    return {
        "source": str(source_dir),
        "status": "loaded",
        "competitions": str(len(competitions)),
        "matches": str(len(matches)),
        "events": str(len(events)),
        "lineups": str(len(lineups)),
    }
