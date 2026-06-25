from pathlib import Path

import pandas as pd


def read_metrica_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_metrica_events(source_dir: Path) -> pd.DataFrame:
    paths = sorted(source_dir.glob("*Events*.csv")) + sorted(source_dir.glob("*events*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = read_metrica_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_metrica_tracking(source_dir: Path) -> pd.DataFrame:
    paths = sorted(source_dir.glob("*Tracking*.csv")) + sorted(source_dir.glob("*tracking*.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = read_metrica_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ingest_metrica(source_dir: Path) -> dict[str, str]:
    events = load_metrica_events(source_dir)
    tracking = load_metrica_tracking(source_dir)
    return {
        "source": str(source_dir),
        "status": "loaded",
        "events": str(len(events)),
        "tracking_rows": str(len(tracking)),
    }
