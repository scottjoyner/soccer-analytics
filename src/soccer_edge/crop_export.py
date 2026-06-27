from pathlib import Path

import pandas as pd

from soccer_edge.media_reader import require_media_reader


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(int(round(min(x1, x2))), width - 1))
    top = max(0, min(int(round(min(y1, y2))), height - 1))
    right = max(left + 1, min(int(round(max(x1, x2))), width))
    bottom = max(top + 1, min(int(round(max(y1, y2))), height))
    return left, top, right, bottom


def crop_filename(row: pd.Series, index: int, frame_column: str = "frame_idx", class_column: str = "class_name") -> str:
    frame = row.get(frame_column, index)
    class_name = str(row.get(class_column, "object")).replace("/", "_").replace(" ", "_")
    return f"{frame}_{index}_{class_name}.jpg"


def export_image_crops(
    detections: pd.DataFrame,
    output_dir: Path,
    image_path_column: str = "image_path",
) -> pd.DataFrame:
    if image_path_column not in detections.columns:
        raise ValueError(f"missing image path column: {image_path_column}")
    reader = require_media_reader()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for idx, row in detections.reset_index(drop=True).iterrows():
        image_path = Path(str(row[image_path_column]))
        image = reader.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        height, width = image.shape[:2]
        left, top, right, bottom = clamp_box(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), width, height)
        crop = image[top:bottom, left:right]
        crop_path = output_dir / crop_filename(row, idx)
        reader.imwrite(str(crop_path), crop)
        output_row = row.to_dict()
        output_row["crop_path"] = str(crop_path)
        output_row["crop_x1"] = left
        output_row["crop_y1"] = top
        output_row["crop_x2"] = right
        output_row["crop_y2"] = bottom
        rows.append(output_row)
    return pd.DataFrame(rows)


def export_image_crops_from_table(
    source: Path,
    output_dir: Path,
    manifest_output: Path,
    image_path_column: str = "image_path",
) -> Path:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    manifest = export_image_crops(frame, output_dir=output_dir, image_path_column=image_path_column)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False)
    return manifest_output
