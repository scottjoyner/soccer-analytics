from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnnotationDatasetConfig:
    root: Path
    train_images: Path
    val_images: Path
    class_names: list[str]
    test_images: Path | None = None


def yaml_list(values: list[str]) -> str:
    return "\n".join(f"  {idx}: {name}" for idx, name in enumerate(values))


def annotation_dataset_yaml(config: AnnotationDatasetConfig) -> str:
    lines = [
        f"path: {config.root}",
        f"train: {config.train_images}",
        f"val: {config.val_images}",
    ]
    if config.test_images is not None:
        lines.append(f"test: {config.test_images}")
    lines.extend(["names:", yaml_list(config.class_names), ""])
    return "\n".join(lines)


def write_annotation_dataset_config(config: AnnotationDatasetConfig, output_path: Path) -> Path:
    if not config.class_names:
        raise ValueError("class_names cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(annotation_dataset_yaml(config), encoding="utf-8")
    return output_path


def write_annotation_dataset_config_from_values(
    root: Path,
    train_images: Path,
    val_images: Path,
    class_names: list[str],
    output_path: Path,
    test_images: Path | None = None,
) -> Path:
    return write_annotation_dataset_config(
        AnnotationDatasetConfig(
            root=root,
            train_images=train_images,
            val_images=val_images,
            class_names=class_names,
            test_images=test_images,
        ),
        output_path,
    )
