import json
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_soccernet_json_files(source_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(source_dir.rglob("*.json")):
        data = read_json(path)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    row = dict(item)
                    row["source_file"] = str(path.relative_to(source_dir))
                    rows.append(row)
        elif isinstance(data, dict):
            row = dict(data)
            row["source_file"] = str(path.relative_to(source_dir))
            rows.append(row)
    return pd.DataFrame(rows)


def load_soccernet_csv_files(source_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(source_dir.rglob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path.relative_to(source_dir))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ingest_soccernet(source_dir: Path) -> dict[str, str]:
    json_rows = load_soccernet_json_files(source_dir)
    csv_rows = load_soccernet_csv_files(source_dir)
    return {
        "source": str(source_dir),
        "status": "loaded",
        "json_rows": str(len(json_rows)),
        "csv_rows": str(len(csv_rows)),
    }
