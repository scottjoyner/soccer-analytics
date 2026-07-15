import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app
from soccer_edge.ingest.football_data_loader import load_football_data, normalize_football_data_frame
from soccer_edge.ingest.openfootball_loader import load_openfootball, normalize_openfootball_frame, parse_score
from soccer_edge.ingest.processed_tables import write_football_data_processed, write_openfootball_processed

runner = CliRunner()

OPENFOOTBALL_CSV = (
    "date,home_team,away_team,score,competition,season\n"
    "2023-08-11,Arsenal,Manchester City,2-1,Premier League,2023/24\n"
    "2023-08-12,Liverpool,Chelsea,1-1,Premier League,2023/24\n"
    "2023-09-01,Everton,Fulham,,Premier League,2023/24\n"
)

FOOTBALL_DATA_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Season\n"
    "E0,11/08/2023,Arsenal,Manchester City,2,1,H,2023/24\n"
    "E0,12/08/2023,Liverpool,Chelsea,1,1,D,2023/24\n"
)


def test_parse_score() -> None:
    assert parse_score("2-1") == (2, 1)
    assert parse_score("0-0") == (0, 0)
    assert parse_score("") == (None, None)
    assert parse_score(None) == (None, None)
    assert parse_score("2") == (None, None)


def test_load_openfootball(tmp_path) -> None:
    source = tmp_path / "openfootball"
    source.mkdir()
    (source / "sample.csv").write_text(OPENFOOTBALL_CSV, encoding="utf-8")
    frame = load_openfootball(source)
    assert len(frame) == 3
    row = frame.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_score"] == 1
    assert row["result"] == "H"
    assert row["season"] == "2023/24"
    missing = frame.iloc[2]
    assert pd.isna(missing["home_score"])
    assert pd.isna(missing["result"])


def test_normalize_openfootball_columns(tmp_path) -> None:
    source = tmp_path / "openfootball"
    source.mkdir()
    (source / "sample.csv").write_text(OPENFOOTBALL_CSV, encoding="utf-8")
    normalized = normalize_openfootball_frame(pd.read_csv(source / "sample.csv"), source_path=source / "sample.csv")
    for column in ("match_id", "match_date", "home_team", "away_team", "home_score", "away_score", "competition", "season", "result"):
        assert column in normalized.columns


def test_load_football_data(tmp_path) -> None:
    source = tmp_path / "football_data"
    source.mkdir()
    (source / "sample.csv").write_text(FOOTBALL_DATA_CSV, encoding="utf-8")
    frame = load_football_data(source)
    assert len(frame) == 2
    row = frame.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["home_score"] == 2
    assert row["away_score"] == 1
    assert row["competition"] == "E0"
    assert row["result"] == "H"
    assert row["season"] == "2023/24"


def test_load_football_data_unknown_columns(tmp_path) -> None:
    frame = pd.DataFrame([{"X": 1}])
    normalized = normalize_football_data_frame(frame)
    assert len(normalized) == 1
    assert normalized.iloc[0]["home_team"] == ""


def test_write_openfootball_processed(tmp_path) -> None:
    source = tmp_path / "openfootball"
    source.mkdir()
    (source / "sample.csv").write_text(OPENFOOTBALL_CSV, encoding="utf-8")
    output = tmp_path / "output"
    paths = write_openfootball_processed(source, output, dataset_version="v1")
    assert paths["openfootball_matches"].exists()
    table = pd.read_parquet(paths["openfootball_matches"])
    assert table["lineage_source_name"].iloc[0] == "openfootball"
    assert table["lineage_dataset_version"].iloc[0] == "v1"
    assert table["home_score"].iloc[0] == 2


def test_write_football_data_processed(tmp_path) -> None:
    source = tmp_path / "football_data"
    source.mkdir()
    (source / "sample.csv").write_text(FOOTBALL_DATA_CSV, encoding="utf-8")
    output = tmp_path / "output"
    paths = write_football_data_processed(source, output, dataset_version="v2")
    assert paths["football_data_matches"].exists()
    table = pd.read_parquet(paths["football_data_matches"])
    assert table["lineage_source_name"].iloc[0] == "football-data"
    assert table["lineage_dataset_version"].iloc[0] == "v2"
    assert table["away_score"].iloc[1] == 1


def test_cli_openfootball_and_football_data_commands(tmp_path) -> None:
    of_source = tmp_path / "openfootball"
    of_source.mkdir()
    (of_source / "sample.csv").write_text(OPENFOOTBALL_CSV, encoding="utf-8")
    fd_source = tmp_path / "football_data"
    fd_source.mkdir()
    (fd_source / "sample.csv").write_text(FOOTBALL_DATA_CSV, encoding="utf-8")

    result = runner.invoke(app, ["ingest", "openfootball", "--path", str(of_source)])
    assert result.exit_code == 0
    assert "loaded" in result.stdout

    result = runner.invoke(app, ["ingest", "football-data", "--path", str(fd_source)])
    assert result.exit_code == 0
    assert "loaded" in result.stdout


def test_cli_write_processed_open_data(tmp_path) -> None:
    of_source = tmp_path / "openfootball"
    of_source.mkdir()
    (of_source / "sample.csv").write_text(OPENFOOTBALL_CSV, encoding="utf-8")
    fd_source = tmp_path / "football_data"
    fd_source.mkdir()
    (fd_source / "sample.csv").write_text(FOOTBALL_DATA_CSV, encoding="utf-8")

    of_output = tmp_path / "of_output"
    result = runner.invoke(
        app,
        ["ingest", "write-processed", "--source", str(of_source), "--output-dir", str(of_output), "--source-type", "openfootball"],
    )
    assert result.exit_code == 0
    assert (of_output / "openfootball_matches.parquet").exists()

    fd_output = tmp_path / "fd_output"
    result = runner.invoke(
        app,
        ["ingest", "write-processed", "--source", str(fd_source), "--output-dir", str(fd_output), "--source-type", "football-data"],
    )
    assert result.exit_code == 0
    assert (fd_output / "football_data_matches.parquet").exists()
