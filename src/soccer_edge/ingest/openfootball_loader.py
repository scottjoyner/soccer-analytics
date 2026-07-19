import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_DATE_COLUMNS = ("date", "match_date", "day")
_HOME_COLUMNS = ("home_team", "home", "team1")
_AWAY_COLUMNS = ("away_team", "away", "team2")
_SCORE_COLUMNS = ("score", "result", "ft_score")
_COMPETITION_COLUMNS = ("competition", "league", "comp", "tournament", "name")
_SEASON_COLUMNS = ("season", "year", "campaign")


def _first_present(candidates: tuple[str, ...], columns) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _slug(value: object) -> str:
    return re.sub(r"\W+", "_", str(value).lower()).strip("_")


def _to_str(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return str(raw).strip()


def _score_from_mapping(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    for key in ("ft", "full_time", "fulltime"):
        value = raw.get(key)
        if isinstance(value, list | tuple) and len(value) >= 2:
            return _coerce_int(value[0]), _coerce_int(value[1])
        if isinstance(value, dict):
            return _coerce_int(value.get("home") or value.get("team1")), _coerce_int(value.get("away") or value.get("team2"))
    return _coerce_int(raw.get("home") or raw.get("team1")), _coerce_int(raw.get("away") or raw.get("team2"))


def _coerce_int(raw: object) -> int | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_score(raw: object) -> tuple[int | None, int | None]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    if isinstance(raw, dict):
        return _score_from_mapping(raw)
    if isinstance(raw, list | tuple) and len(raw) >= 2:
        return _coerce_int(raw[0]), _coerce_int(raw[1])
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None, None
    if "-" in text:
        home_part, away_part = text.split("-", 1)
        return _coerce_int(home_part), _coerce_int(away_part)
    return None, None


def _derive_result(home_score: int | None, away_score: int | None) -> str | None:
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _team_name(raw: object) -> str:
    if isinstance(raw, dict):
        return _to_str(raw.get("name") or raw.get("title") or raw.get("code") or raw.get("id"))
    return _to_str(raw)


def normalize_openfootball_frame(frame: pd.DataFrame, source_path: Path | str | None = None) -> pd.DataFrame:
    columns = list(frame.columns)
    date_col = _first_present(_DATE_COLUMNS, columns)
    home_col = _first_present(_HOME_COLUMNS, columns)
    away_col = _first_present(_AWAY_COLUMNS, columns)
    score_col = _first_present(_SCORE_COLUMNS, columns)
    competition_col = _first_present(_COMPETITION_COLUMNS, columns)
    season_col = _first_present(_SEASON_COLUMNS, columns)

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        home_score, away_score = parse_score(row[score_col]) if score_col is not None else (None, None)
        match_date = _to_str(row[date_col]) if date_col is not None else ""
        home_team = _team_name(row[home_col]) if home_col is not None else ""
        away_team = _team_name(row[away_col]) if away_col is not None else ""
        competition = _to_str(row[competition_col]) if competition_col is not None else ""
        season = _to_str(row[season_col]) if season_col is not None else ""
        match_id = f"openfootball_{_slug(match_date)}_{_slug(home_team)}_{_slug(away_team)}"
        rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "competition": competition,
                "season": season,
                "result": _derive_result(home_score, away_score),
                "source_file": str(source_path) if source_path is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _json_matches_to_rows(data: object, source_path: Path) -> list[dict[str, object]]:
    if isinstance(data, dict):
        matches = data.get("matches") or data.get("games") or []
        competition = _to_str(data.get("name") or data.get("competition") or data.get("league"))
        season = _to_str(data.get("season") or data.get("year"))
    elif isinstance(data, list):
        matches = data
        competition = ""
        season = ""
    else:
        return []

    rows: list[dict[str, object]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        home_team = _team_name(match.get("team1") or match.get("home") or match.get("home_team"))
        away_team = _team_name(match.get("team2") or match.get("away") or match.get("away_team"))
        match_date = _to_str(match.get("date") or match.get("match_date") or match.get("day"))
        home_score, away_score = parse_score(match.get("score") or match.get("result") or match.get("ft_score"))
        row_competition = _to_str(match.get("competition") or match.get("league") or competition)
        row_season = _to_str(match.get("season") or match.get("year") or season)
        match_id = f"openfootball_{_slug(match_date)}_{_slug(home_team)}_{_slug(away_team)}"
        rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "competition": row_competition,
                "season": row_season,
                "result": _derive_result(home_score, away_score),
                "source_file": str(source_path),
            }
        )
    return rows


def load_openfootball(source_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(source_dir.rglob("*.csv"))
    json_paths = sorted(source_dir.rglob("*.json"))
    frames: list[pd.DataFrame] = []
    frames.extend(normalize_openfootball_frame(pd.read_csv(path), source_path=path) for path in csv_paths)
    for path in json_paths:
        with path.open(encoding="utf-8") as handle:
            rows = _json_matches_to_rows(json.load(handle), path)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ingest_openfootball(source_dir: Path) -> dict[str, str]:
    frame = load_openfootball(source_dir)
    return {
        "source": str(source_dir),
        "status": "loaded",
        "rows": str(len(frame)),
    }
