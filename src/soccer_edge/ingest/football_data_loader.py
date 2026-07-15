import re
from pathlib import Path

import pandas as pd

_DATE_COLUMNS = ("Date", "date", "match_date")
_HOME_COLUMNS = ("HomeTeam", "home_team")
_AWAY_COLUMNS = ("AwayTeam", "away_team")
_HOME_SCORE_COLUMNS = ("FTHG", "home_score")
_AWAY_SCORE_COLUMNS = ("FTAG", "away_score")
_RESULT_COLUMNS = ("FTR", "result")
_COMPETITION_COLUMNS = ("Div", "competition", "league")
_SEASON_COLUMNS = ("Season", "season", "year")


def _first_present(candidates: tuple[str, ...], columns) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


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


def _to_str(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return str(raw).strip()


def normalize_football_data_frame(frame: pd.DataFrame, source_path: Path | str | None = None) -> pd.DataFrame:
    columns = list(frame.columns)
    date_col = _first_present(_DATE_COLUMNS, columns)
    home_col = _first_present(_HOME_COLUMNS, columns)
    away_col = _first_present(_AWAY_COLUMNS, columns)
    home_score_col = _first_present(_HOME_SCORE_COLUMNS, columns)
    away_score_col = _first_present(_AWAY_SCORE_COLUMNS, columns)
    result_col = _first_present(_RESULT_COLUMNS, columns)
    competition_col = _first_present(_COMPETITION_COLUMNS, columns)
    season_col = _first_present(_SEASON_COLUMNS, columns)

    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        home_score = _coerce_int(row[home_score_col]) if home_score_col is not None else None
        away_score = _coerce_int(row[away_score_col]) if away_score_col is not None else None
        match_date = _to_str(row[date_col]) if date_col is not None else ""
        home_team = _to_str(row[home_col]) if home_col is not None else ""
        away_team = _to_str(row[away_col]) if away_col is not None else ""
        result = _to_str(row[result_col]) if result_col is not None else ""
        competition = _to_str(row[competition_col]) if competition_col is not None else ""
        season = _to_str(row[season_col]) if season_col is not None else ""
        match_id = f"football_data_{_slug(match_date)}_{_slug(competition)}_{_slug(home_team)}_{_slug(away_team)}"
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
                "result": result,
                "source_file": str(source_path) if source_path is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _slug(value: object) -> str:
    return re.sub(r"\W+", "_", str(value).lower()).strip("_")


def load_football_data(source_dir: Path) -> pd.DataFrame:
    paths = sorted(source_dir.rglob("*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = [normalize_football_data_frame(pd.read_csv(path), source_path=path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def ingest_football_data(source_dir: Path) -> dict[str, str]:
    frame = load_football_data(source_dir)
    return {
        "source": str(source_dir),
        "status": "loaded",
        "rows": str(len(frame)),
    }
