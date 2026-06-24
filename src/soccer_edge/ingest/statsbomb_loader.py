from pathlib import Path


def ingest_statsbomb(source_dir: Path) -> dict[str, str]:
    return {
        "source": str(source_dir),
        "status": "not_implemented",
        "message": "StatsBomb loader placeholder."
    }
