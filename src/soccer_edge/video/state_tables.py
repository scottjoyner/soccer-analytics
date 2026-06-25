from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from soccer_edge.store.table_store import save_table


def rows_to_dataframe(rows: Iterable[object]) -> pd.DataFrame:
    materialized = list(rows)
    if not materialized:
        return pd.DataFrame()
    return pd.DataFrame([asdict(row) if is_dataclass(row) else row for row in materialized])


def write_state_table(rows: Iterable[object], destination: Path) -> None:
    save_table(rows_to_dataframe(rows), destination)


def write_video_state_tables(
    output_dir: Path,
    detections: Iterable[object] = (),
    tracks: Iterable[object] = (),
    ball_states: Iterable[object] = (),
    player_states: Iterable[object] = (),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detections": output_dir / "detections.parquet",
        "tracks": output_dir / "tracks.parquet",
        "ball_states": output_dir / "ball_states.parquet",
        "player_states": output_dir / "player_states.parquet",
    }
    write_state_table(detections, paths["detections"])
    write_state_table(tracks, paths["tracks"])
    write_state_table(ball_states, paths["ball_states"])
    write_state_table(player_states, paths["player_states"])
    return paths
