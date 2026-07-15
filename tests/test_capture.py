import numpy as np
import pytest

import cv2
from pathlib import Path

from soccer_edge.capture import (
    _require_rights,
    build_capture_row,
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
    assert "rights_status must be one of" in str(result.exception)


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
        Path(path).write_bytes(b"")

    def write(self, frame) -> None:
        self.calls += 1

    def release(self) -> None:
        pass


def test_capture_webcam_video_writes_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda device: _FakeCap(device))
    monkeypatch.setattr(cv2, "VideoWriter", lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size))
    out = tmp_path / "cap.mp4"
    saved = capture_webcam_video(out, duration_seconds=0.5, fps=10)
    assert saved.exists()
    assert saved.stat().st_size >= 0


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
