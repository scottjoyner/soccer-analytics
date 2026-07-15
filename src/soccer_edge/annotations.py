from pathlib import Path

import pandas as pd
import shutil

from soccer_edge.annotation_dataset import write_annotation_dataset_config_from_values


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
    known = detections[detections[class_column].isin(classes)]
    paths: dict[str, Path] = {}
    for frame_id, group in known.groupby(frame_column):
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


def arrange_yolo_dataset(
    detections: pd.DataFrame,
    output_dir: Path,
    classes: list[str],
    image_width: float,
    image_height: float,
    train_fraction: float = 0.8,
    group_column: str = "frame_idx",
    image_column: str = "image_path",
    class_column: str = "class_name",
    link_images: bool = True,
) -> dict[str, Path]:
    """Arrange detection rows into a YOLO dataset layout for ultralytics training.

    Produces ``images/{train,val}`` and ``labels/{train,val}`` directories, plus a
    ``data.yaml`` config and ``classes.txt``. Detections from the same frame group are
    never split across train/val so a frame's boxes stay together.
    """

    output_dir = Path(output_dir)
    class_to_id = build_class_index(classes)
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    for directory in (images_train, images_val, labels_train, labels_val):
        directory.mkdir(parents=True, exist_ok=True)

    known = detections[detections[class_column].isin(classes)]
    if known.empty:
        raise ValueError("no detections matched the provided classes")
    groups = sorted(known[group_column].dropna().unique().tolist())
    if not groups:
        raise ValueError(f"no non-null values in group column {group_column!r}")
    n_train = max(1, int(round(len(groups) * train_fraction)))
    if len(groups) > 1:
        n_train = min(n_train, len(groups) - 1)
    train_groups = set(groups[:n_train])

    has_images = image_column in known.columns
    for frame_id, group in known.groupby(group_column):
        split = "train" if frame_id in train_groups else "val"
        label_dir = labels_train if split == "train" else labels_val
        image_dir = images_train if split == "train" else images_val
        label_path = label_dir / f"{frame_id}.txt"
        lines = [
            normalized_box_line(row, class_to_id, image_width, image_height, class_column)
            for _, row in group.iterrows()
        ]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if has_images:
            image_src = group.iloc[0][image_column]
            if pd.notna(image_src) and str(image_src):
                src = Path(str(image_src))
                if src.exists():
                    dst = image_dir / f"{frame_id}{src.suffix}"
                    if link_images:
                        if not dst.exists():
                            dst.symlink_to(src.resolve())
                    else:
                        shutil.copyfile(src, dst)

    (output_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    config_path = write_annotation_dataset_config_from_values(
        root=output_dir,
        train_images=Path("images/train"),
        val_images=Path("images/val"),
        class_names=classes,
        output_path=output_dir / "data.yaml",
    )
    return {
        "images_train": images_train,
        "images_val": images_val,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "classes": output_dir / "classes.txt",
        "config": config_path,
    }


def arrange_yolo_dataset_from_table(
    source: Path,
    output_dir: Path,
    classes: list[str],
    image_width: float,
    image_height: float,
    train_fraction: float = 0.8,
    group_column: str = "frame_idx",
    image_column: str = "image_path",
    class_column: str = "class_name",
    link_images: bool = True,
) -> dict[str, Path]:
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    return arrange_yolo_dataset(
        frame,
        output_dir,
        classes,
        image_width,
        image_height,
        train_fraction=train_fraction,
        group_column=group_column,
        image_column=image_column,
        class_column=class_column,
        link_images=link_images,
    )
