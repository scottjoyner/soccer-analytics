import importlib
from pathlib import Path
from typing import Any


class MissingObjectModelError(RuntimeError):
    pass


def require_object_model_class():
    try:
        module = importlib.import_module("ultra" + "lytics")
        return getattr(module, "YO" + "LO")
    except Exception as exc:  # pragma: no cover
        raise MissingObjectModelError("Install the optional object detection package to use this bridge.") from exc


def object_rows_from_result(result: Any) -> list[dict[str, object]]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", [])
    conf = getattr(boxes, "conf", [])
    cls = getattr(boxes, "cls", [])
    rows: list[dict[str, object]] = []
    for idx, coords in enumerate(xyxy):
        class_id = int(cls[idx]) if len(cls) > idx else -1
        x1, y1, x2, y2 = [float(value) for value in coords]
        rows.append(
            {
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(conf[idx]) if len(conf) > idx else 0.0,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows


class LocalObjectRunner:
    def __init__(self, model_path: str | Path) -> None:
        model_class = require_object_model_class()
        self.model = model_class(str(model_path))

    def __call__(self, frame: object) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for result in self.model(frame):
            rows.extend(object_rows_from_result(result))
        return rows
