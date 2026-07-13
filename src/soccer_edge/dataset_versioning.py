import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetAssetVersion:
    path: str
    sha256: str
    size_bytes: int
    row_count: int | None = None
    column_count: int | None = None


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_shape(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    except Exception:
        return None, None
    return len(frame), len(frame.columns)


def asset_version(path: Path) -> DatasetAssetVersion:
    rows, columns = table_shape(path)
    return DatasetAssetVersion(
        path=str(path),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
        row_count=rows,
        column_count=columns,
    )


def dataset_versions(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame([asdict(asset_version(path)) for path in paths])


def dataset_version_id(paths: list[Path], prefix: str = "ds") -> str:
    versions = sorted((asset_version(path) for path in paths), key=lambda item: item.path)
    digest = hashlib.sha256()
    for version in versions:
        digest.update(version.path.encode("utf-8"))
        digest.update(version.sha256.encode("utf-8"))
        digest.update(str(version.size_bytes).encode("utf-8"))
        digest.update(str(version.row_count).encode("utf-8"))
        digest.update(str(version.column_count).encode("utf-8"))
    return f"{prefix}_{digest.hexdigest()[:16]}"


def write_dataset_versions(paths: list[Path], output: Path) -> Path:
    frame = dataset_versions(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
    return output
