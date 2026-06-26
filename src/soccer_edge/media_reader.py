import importlib


class MissingMediaReaderError(RuntimeError):
    pass


def require_media_reader():
    try:
        return importlib.import_module("cv2")
    except Exception as exc:  # pragma: no cover
        raise MissingMediaReaderError("Install opencv-python to read local frame data.") from exc
