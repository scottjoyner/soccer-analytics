import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunMetadata:
    name: str
    version: str
    created_at_utc: str
    feature_names: list[str]
    metrics: dict[str, float]
    notes: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_metadata(metadata: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")


def read_run_metadata(path: Path) -> RunMetadata:
    return RunMetadata(**json.loads(path.read_text(encoding="utf-8")))
