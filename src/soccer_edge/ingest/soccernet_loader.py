from pathlib import Path


def ingest_soccernet(source_dir: Path) -> dict[str, str]:
    return {
        "source": str(source_dir),
        "status": "not_implemented",
        "message": "SoccerNet loader placeholder."
    }
