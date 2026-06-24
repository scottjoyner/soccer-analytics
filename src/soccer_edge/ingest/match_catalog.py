import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatchCatalogRow:
    match_id: str
    competition: str
    season: str
    match_date: str
    stage: str
    home_team: str
    away_team: str
    venue: str
    home_score: str
    away_score: str
    status: str


def match_catalog_row_from_dict(row: dict[str, str]) -> MatchCatalogRow:
    return MatchCatalogRow(
        match_id=row["match_id"],
        competition=row.get("competition", ""),
        season=row.get("season", ""),
        match_date=row.get("match_date", ""),
        stage=row.get("stage", ""),
        home_team=row.get("home_team", ""),
        away_team=row.get("away_team", ""),
        venue=row.get("venue", ""),
        home_score=row.get("home_score", ""),
        away_score=row.get("away_score", ""),
        status=row.get("status", "scheduled"),
    )


def read_match_catalog(path: Path) -> list[MatchCatalogRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [match_catalog_row_from_dict(row) for row in reader]
