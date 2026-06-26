from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.state_tables import write_video_state_tables


@dataclass(frozen=True)
class MediaRunResult:
    input_path: Path
    output_dir: Path
    frame_count: int
    table_paths: dict[str, Path]


def run_media_table_stub(input_path: Path, output_dir: Path, frame_count: int = 0) -> MediaRunResult:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    table_paths = write_video_state_tables(output_dir=output_dir)
    return MediaRunResult(
        input_path=input_path,
        output_dir=output_dir,
        frame_count=frame_count,
        table_paths=table_paths,
    )
