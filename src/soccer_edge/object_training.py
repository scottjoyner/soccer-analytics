import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from soccer_edge.object_model import require_object_model_class


@dataclass(frozen=True)
class ObjectTrainingConfig:
    data_config: Path
    base_model: Path
    output_dir: Path
    run_name: str = "local_object_model"
    epochs: int = 50
    image_size: int = 640


def object_training_kwargs(config: ObjectTrainingConfig) -> dict[str, Any]:
    return {
        "data": str(config.data_config),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "project": str(config.output_dir),
        "name": config.run_name,
    }


def write_object_training_config(config: ObjectTrainingConfig) -> Path:
    path = config.output_dir / "object_training_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: str(value) for key, value in asdict(config).items()}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_object_training(config: ObjectTrainingConfig) -> dict[str, Path]:
    model_class = require_object_model_class()
    config_path = write_object_training_config(config)
    model = model_class(str(config.base_model))
    model.train(**object_training_kwargs(config))
    return {"config": config_path, "run_dir": config.output_dir / config.run_name}
