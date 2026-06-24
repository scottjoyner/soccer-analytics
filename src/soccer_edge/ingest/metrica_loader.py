from pathlib import Path


def ingest_metrica(source_dir: Path) -> dict[str, str]:
    return {
        "source": str(source_dir),
        "status": "not_implemented",
        "message": "Metrica loader placeholder."
    }
