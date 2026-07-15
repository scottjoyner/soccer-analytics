import re
from pathlib import Path

import pandas as pd

_DATE_COLUMNS = ("date", "match_date", "day")
_HOME_COLUMNS = ("home_team", "home", "team1")
_AWAY_COLUMNS = ("away_team", "away", "team2")
_SCORE_COLUMNS = ("score", "result", "ft_score")
_COMPETITION_COLUMNS = ("competition", "league", "comp", "tournament")
_SEASON_COLUMNS = ("season", "year", "campaign")


def _first_present(candidates: tuple[str, ...], columns) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _slug(value: object) -> str:
    return re.sub(r"\W+", "_", str(value).lower()).strip("_")


def parse_score(raw: object) -> tuple[int | None, int | None]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", ""}:
        return None, None
    if "-" in text:
        home_part, away_part = text.split("-", 1)
        try:
            return int(home_part.strip()), int(away_part.strip())
        except ValueError:
            return None, None
    return None, None


def _derive_result(home_score: int | None, away_score: int | None) -> str | None:
    if home_score is None or away_score is None:
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def normalize_openfootball_frame(frame: pd.DataFrame, source_path: Path | str | None = None) -> pd.DataFrame:
    columns = list(frame.columns)
    date_col = _first_present(_DATE_COLUMNS, columns)
    home_col = _first_present(_HOME_COLUMNS, columns)
    away_col = _first_present(_AWAY_COLUMNS, columns)
    score_col = _first_present(_SCORE_COLUMNS, columns)
    competition_col = _first_present(_COMPETITION_COLUMNS, columns)
    season_col = _first_present(_SEASON_COLUMNS, columns)

    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        home_score, away_score = parse_score(row[score_col]) if score_col is not None else (None, None)
        match_date = str(row[date_col]) if date_col is not None else ""
        home_team = str(row[home_col]) if home_col is not None else ""
        away_team = str(row[away_col]) if away_col is not None else ""
        match_id = f"openfootball_{_slug(match_date)}_{_slug(home_team)}_{_slug(away_team)}"
        rows.append(
            {
                "match_id": match_id,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "competition": str(row[competition_col]) if competition_col is not None else "",
                "season": str(row[season_col]) if season_col is not None else "",
                "result": _derive_result(home_score, away_score),
                "source_file": str(source_path) if source_path is not None else "",
            }
        )
    return pd.DataFrame(rows)


def load_openfootball(source_dir: Path) -> pd.DataFrame:
    paths = sorted(source_dir.rglob("*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = [normalize_openfootball_frame(pd.read_csv(path), source_path=path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def ingest_openfootball(source_dir: Path) -> dict[str, str]:
    frame = load_openfootball(source_dir)
    return {
        "source": str(source_dir),
        "status": "loaded",
        "rows": str(len(frame)),
    }
