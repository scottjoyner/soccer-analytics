import numpy as np
import pandas as pd
import pytest

import cv2
from pathlib import Path

from soccer_edge.capture import (
    _require_rights,
    build_capture_row,
    capture_and_detect,
    capture_and_register,
    capture_screen_image,
    capture_webcam_video,
    parse_region,
)
from soccer_edge.video.manifest import (
    VideoManifestRow,
    append_manifest_row,
    read_video_manifest,
    validate_processable_video,
)

from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_capture_cli_refuses_unowned() -> None:
    result = runner.invoke(
        app,
        ["capture", "screen", "--rights-status", "pending", "--rights-reference", "ref"],
    )
    assert result.exit_code != 0
    assert "rights_status must be one of" in (str(result.exception) or result.output)


def test_require_rights_rejects_unowned() -> None:
    with pytest.raises(ValueError):
        _require_rights("pending", "ref")
    with pytest.raises(ValueError):
        _require_rights("owned", "")
    _require_rights("owned", "ref1")  # does not raise


def test_parse_region() -> None:
    assert parse_region(None) is None
    assert parse_region("1,2,3,4") == {"left": 1, "top": 2, "width": 3, "height": 4}
    with pytest.raises(ValueError):
        parse_region("1,2,3")


def test_build_capture_row_is_processable() -> None:
    row = build_capture_row(
        video_id="v1",
        local_path="/tmp/x.mp4",
        rights_status="owned",
        rights_reference="ref1",
        capture_source="screen",
    )
    assert row.source_url == "capture://screen"
    assert row.is_processable


class _FakeCap:
    def __init__(self, device: int) -> None:
        self.device = device
        self._frames = 3
        self._i = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> int:
        return 640 if prop == cv2.CAP_PROP_FRAME_WIDTH else 480

    def read(self):
        if self._i >= self._frames:
            return False, None
        self._i += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _FakeWriter:
    def __init__(self, path, fourcc, fps, size) -> None:
        self.path = path
        self.size = size
        self.calls = 0
        self.released = False
        Path(path).write_bytes(b"")

    def write(self, frame) -> None:
        self.calls += 1

    def release(self) -> None:
        self.released = True


def test_capture_webcam_video_writes_frames(tmp_path, monkeypatch) -> None:
    writers = []

    def _make_writer(path, fourcc, fps, size):
        w = _FakeWriter(path, fourcc, fps, size)
        writers.append(w)
        return w

    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _FakeCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", _make_writer)
    out = tmp_path / "cap.mp4"
    saved = capture_webcam_video(out, duration_seconds=0.5, fps=10)
    assert saved.exists()
    assert writers[0].calls > 0
    assert writers[0].released


class _FakeShot:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.size = (width, height)
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8).tobytes()


class _FakeSct:
    def __init__(self) -> None:
        self.monitors = [{"left": 0, "top": 0, "width": 100, "height": 80}]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def grab(self, area):
        return _FakeShot(area.get("width", 100), area.get("height", 80))

    def shot(self, mon: int = 1, output: str = "out.png") -> None:
        Path(output).write_bytes(b"fake-png")


class _FakeMss:
    def mss(self):
        return _FakeSct()


def test_capture_screen_video_registers_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    out = tmp_path / "screen.mp4"
    manifest = tmp_path / "manifests" / "video_manifest.csv"
    saved, row = capture_and_register(
        "screen",
        out,
        duration_seconds=0.5,
        fps=10,
        manifest_path=manifest,
        rights_status="licensed",
        rights_reference="license-abc",
        video_id="cap1",
        mss_module=_FakeMss(),
    )
    assert saved.exists()
    assert row.source_url == "capture://screen"
    rows = read_video_manifest(manifest)
    assert len(rows) == 1
    assert rows[0].video_id == "cap1"


def test_capture_screen_image(tmp_path) -> None:
    out = tmp_path / "shot.png"
    saved = capture_screen_image(out, mss_module=_FakeMss())
    assert saved.exists()


def test_manifest_row_passes_rights_gate(tmp_path) -> None:
    licensed = tmp_path / "licensed"
    captures = licensed / "captures"
    captures.mkdir(parents=True)
    capture_file = captures / "screen_X.mp4"
    capture_file.write_bytes(b"vid")
    manifest = tmp_path / "manifests" / "video_manifest.csv"
    row = build_capture_row(
        video_id="v1",
        local_path=capture_file,
        rights_status="owned",
        rights_reference="ref1",
        capture_source="screen",
    )
    append_manifest_row(manifest, row)
    rows = read_video_manifest(manifest)
    validate_processable_video(rows[0], licensed)  # does not raise

    pending = VideoManifestRow(
        video_id="v2",
        match_id="",
        competition="",
        season="",
        home_team="",
        away_team="",
        clip_type="training_clip",
        source_url="capture://screen",
        local_path=capture_file,
        period="",
        start_match_second=None,
        end_match_second=None,
        rights_status="pending",
        rights_reference="ref1",
    )
    with pytest.raises(ValueError):
        validate_processable_video(pending, licensed)


class _FakeRunner:
    def __init__(self, model_path=None) -> None:
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return [{"class_name": "person", "confidence": 0.9, "x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}]


def test_capture_and_detect_writes_detections(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _FakeCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    det = tmp_path / "det.csv"
    vid = tmp_path / "cap.mp4"
    result = capture_and_detect(
        "webcam",
        runner=_FakeRunner(),
        duration_seconds=0.2,
        fps=10,
        output_video=vid,
        detections_output=det,
        annotate=True,
    )
    assert result["video"] == vid
    assert det.exists()
    frame = pd.read_csv(det)
    assert list(frame.columns) == ["frame_idx", "timestamp_seconds", "class_name", "confidence", "x1", "y1", "x2", "y2"]
    assert len(frame) == 3
    assert frame.iloc[0]["class_name"] == "person"


def test_capture_cli_detect(tmp_path, monkeypatch) -> None:
    import soccer_edge.object_model as object_model_module

    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _FakeCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    monkeypatch.setattr(object_model_module, "LocalObjectRunner", _FakeRunner)
    manifest = tmp_path / "manifests" / "video_manifest.csv"
    result = runner.invoke(
        app,
        [
            "capture",
            "webcam",
            "--detect",
            "--object-model-path",
            "models/yolov8n.pt",
            "--rights-status",
            "owned",
            "--rights-reference",
            "ref1",
            "--duration",
            "0.2",
            "--detections-output",
            str(tmp_path / "det.csv"),
            "--output",
            str(tmp_path / "cap.mp4"),
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "det.csv").exists()
    assert len(read_video_manifest(manifest)) == 1


class _EmptyCap(_FakeCap):
    def __init__(self, device: int) -> None:
        super().__init__(device)
        self._frames = 0


def test_capture_webcam_zero_frames_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _EmptyCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    with pytest.raises(RuntimeError, match="no frames"):
        capture_webcam_video(tmp_path / "cap.mp4", duration_seconds=0.5, fps=10)


def test_capture_webcam_zero_duration_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _FakeCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    with pytest.raises(ValueError, match="positive"):
        capture_webcam_video(tmp_path / "cap.mp4", duration_seconds=0.0, fps=10)


def test_capture_and_register_zero_frames_no_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _EmptyCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    manifest = tmp_path / "manifests" / "video_manifest.csv"
    with pytest.raises(RuntimeError, match="no frames"):
        capture_and_register(
            "webcam",
            tmp_path / "cap.mp4",
            duration_seconds=0.5,
            fps=10,
            manifest_path=manifest,
            rights_status="owned",
            rights_reference="ref1",
        )
    assert not manifest.exists()


def test_capture_and_detect_zero_frames_no_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _EmptyCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    det = tmp_path / "det.csv"
    result = capture_and_detect(
        "webcam",
        runner=_FakeRunner(),
        duration_seconds=0.2,
        fps=10,
        output_video=tmp_path / "cap.mp4",
        detections_output=det,
    )
    assert result["video"] is None
    assert result["detections"] is None
    assert not det.exists()


def test_capture_and_detect_and_register_zero_frames_no_manifest(tmp_path, monkeypatch) -> None:
    from soccer_edge.capture import capture_and_detect_and_register

    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _EmptyCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    manifest = tmp_path / "manifests" / "video_manifest.csv"
    with pytest.raises(RuntimeError, match="no frames"):
        capture_and_detect_and_register(
            "webcam",
            tmp_path / "cap.mp4",
            runner=_FakeRunner(),
            duration_seconds=0.2,
            fps=10,
            detections_output=tmp_path / "det.csv",
            manifest_path=manifest,
            rights_status="owned",
            rights_reference="ref1",
        )
    assert not manifest.exists()


def test_capture_and_detect_and_register_rejects_image(tmp_path) -> None:
    from soccer_edge.capture import capture_and_detect_and_register

    with pytest.raises(ValueError, match="live detection requires a video source"):
        capture_and_detect_and_register(
            "image",
            tmp_path / "cap.mp4",
            runner=_FakeRunner(),
            detections_output=tmp_path / "det.csv",
            rights_status="owned",
            rights_reference="ref1",
        )
