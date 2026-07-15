from pathlib import Path

import pandas as pd

from soccer_edge.ingest.football_data_loader import load_football_data
from soccer_edge.ingest.lineage import add_lineage_columns
from soccer_edge.ingest.metrica_loader import load_metrica_events, load_metrica_tracking
from soccer_edge.ingest.openfootball_loader import load_openfootball
from soccer_edge.ingest.soccernet_loader import load_soccernet_csv_files, load_soccernet_json_files
from soccer_edge.ingest.statsbomb_loader import load_competitions, load_event_files, load_lineup_files, load_match_files
from soccer_edge.store.table_store import save_table


def write_processed_table(frame: pd.DataFrame, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    save_table(frame, path)
    return path


def write_statsbomb_processed(source_dir: Path, output_dir: Path, dataset_version: str = "unknown") -> dict[str, Path]:
    tables = {
        "statsbomb_competitions": load_competitions(source_dir),
        "statsbomb_matches": load_match_files(source_dir),
        "statsbomb_events": load_event_files(source_dir),
        "statsbomb_lineups": load_lineup_files(source_dir),
    }
    return {
        name: write_processed_table(
            add_lineage_columns(frame, "statsbomb", source_dir, dataset_version), output_dir, name
        )
        for name, frame in tables.items()
    }


def write_metrica_processed(source_dir: Path, output_dir: Path, dataset_version: str = "unknown") -> dict[str, Path]:
    tables = {
        "metrica_events": load_metrica_events(source_dir),
        "metrica_tracking": load_metrica_tracking(source_dir),
    }
    return {
        name: write_processed_table(add_lineage_columns(frame, "metrica", source_dir, dataset_version), output_dir, name)
        for name, frame in tables.items()
    }


def write_soccernet_processed(source_dir: Path, output_dir: Path, dataset_version: str = "unknown") -> dict[str, Path]:
    tables = {
        "soccernet_json": load_soccernet_json_files(source_dir),
        "soccernet_csv": load_soccernet_csv_files(source_dir),
    }
    return {
        name: write_processed_table(add_lineage_columns(frame, "soccernet", source_dir, dataset_version), output_dir, name)
        for name, frame in tables.items()
    }


def write_openfootball_processed(source_dir: Path, output_dir: Path, dataset_version: str = "unknown") -> dict[str, Path]:
    tables = {
        "openfootball_matches": load_openfootball(source_dir),
    }
    return {
        name: write_processed_table(add_lineage_columns(frame, "openfootball", source_dir, dataset_version), output_dir, name)
        for name, frame in tables.items()
    }


def write_football_data_processed(source_dir: Path, output_dir: Path, dataset_version: str = "unknown") -> dict[str, Path]:
    tables = {
        "football_data_matches": load_football_data(source_dir),
    }
    return {
        name: write_processed_table(add_lineage_columns(frame, "football-data", source_dir, dataset_version), output_dir, name)
        for name, frame in tables.items()
    }
