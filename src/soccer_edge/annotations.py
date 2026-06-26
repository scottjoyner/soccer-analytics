from pathlib import Path

import pandas as pd


def normalized_box_line(
    row: pd.Series,
    class_to_id: dict[str, int],
    image_width: float,
    image_height: float,
    class_column: str = "class_name",
) -> str:
    class_name = str(row[class_column])
    class_id = class_to_id[class_name]
    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])
    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = abs(x2 - x1) / image_width
    height = abs(y2 - y1) / image_height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def build_class_index(classes: list[str]) -> dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def write_detection_annotations(
    detections: pd.DataFrame,
    output_dir: Path,
    classes: list[str],
    image_width: float,
    image_height: float,
    frame_column: str = "frame_idx",
    class_column: str = "class_name",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = build_class_index(classes)
    paths: dict[str, Path] = {}
    for frame_id, group in detections.groupby(frame_column):
        path = output_dir / f"{frame_id}.txt"
        lines = [normalized_box_line(row, class_to_id, image_width, image_height, class_column) for _, row in group.iterrows()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[str(frame_id)] = path
    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    paths["classes"] = output_dir / "classes.txt"
    return paths


def write_detection_annotations_from_table(
    source: Path,
    output_dir: Path,
    classes: list[str],
    image_width: float,
    image_height: float,
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    return write_detection_annotations(frame, output_dir, classes, image_width, image_height)
