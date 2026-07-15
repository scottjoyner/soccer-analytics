"""Local screen/webcam/image capture for pipeline intake.

This module records footage from local capture sources (screen, webcam, or a
still screenshot) and registers each capture as a rights-gated manifest row so
the existing licensed-video pipeline can process it.

Safety: capture must only be used for content the operator owns or is
licensed/compatible-license for. Capturing third-party streams (YouTube,
Twitch, etc.) is prohibited by the repository rights policy (AGENTS.md). Every
capture therefore requires an explicit ``rights_status`` from the allowed set
and a recorded ``rights_reference``; capture sources are written with a local
``capture://`` scheme so the modality blocklist still applies.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import cv2

from soccer_edge.video.manifest import PROCESSABLE_RIGHTS_STATUSES, VideoManifestRow

CAPTURE_SCHEME = "capture"
DEFAULT_CAPTURE_DIR = Path("data/raw/video_licensed/captures")
DEFAULT_DURATION_SECONDS = 10.0


def _load_mss():
    try:
        import mss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "screen capture requires the optional 'mss' package; install with 'pip install mss'"
        ) from exc
    return mss


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _require_rights(rights_status: str, rights_reference: str) -> None:
    if rights_status not in PROCESSABLE_RIGHTS_STATUSES:
        raise ValueError(
            f"rights_status must be one of {sorted(PROCESSABLE_RIGHTS_STATUSES)}; "
            "capture of unowned or unapproved content is prohibited by the repo rights policy."
        )
    if not rights_reference:
        raise ValueError(
            "rights_reference is required: explicit written rights must be recorded before capture."
        )


def default_capture_output(capture_source: str, suffix: str) -> Path:
    DEFAULT_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CAPTURE_DIR / f"{capture_source}_{_utc_stamp()}{suffix}"


def parse_region(region: str | None) -> dict | None:
    if not region:
        return None
    parts = [value.strip() for value in region.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be 'left,top,width,height'")
    return {
        "left": int(parts[0]),
        "top": int(parts[1]),
        "width": int(parts[2]),
        "height": int(parts[3]),
    }


def capture_webcam_video(
    output: Path,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    fps: int = 20,
    device: int = 0,
    codec: str = "mp4v",
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames_written = 0
    try:
        for frame, _ in _iter_frames("webcam", duration_seconds=duration_seconds, fps=fps, device=device):
            if writer is None:
                writer = _open_video_writer(output, fps, frame, codec)
            writer.write(frame)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    if frames_written == 0:
        raise RuntimeError(f"webcam capture produced no frames (device {device}); nothing written")
    return output


def capture_screen_video(
    output: Path,
    duration_seconds: float,
    fps: int = 20,
    monitor: int = 1,
    region: dict | None = None,
    codec: str = "mp4v",
    mss_module=None,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames_written = 0
    try:
        for frame, _ in _iter_frames(
            "screen", duration_seconds=duration_seconds, fps=fps, monitor=monitor, region=region, mss_module=mss_module
        ):
            if writer is None:
                writer = _open_video_writer(output, fps, frame, codec)
            writer.write(frame)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()
    if frames_written == 0:
        raise RuntimeError("screen capture produced no frames; nothing written")
    return output


def _iter_frames(
    capture_source: str,
    *,
    duration_seconds: float,
    fps: int = 20,
    monitor: int = 1,
    region: dict | None = None,
    device: int = 0,
    mss_module=None,
):
    start = time.time()
    if capture_source == "webcam":
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open webcam device {device}")
        try:
            while time.time() - start < duration_seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, time.time() - start
        finally:
            cap.release()
    elif capture_source == "screen":
        mss = mss_module or _load_mss()
        with mss.mss() as sct:
            monitors = sct.monitors
            capture_area = region if region is not None else monitors[min(monitor, len(monitors) - 1)]
            while time.time() - start < duration_seconds:
                shot = sct.grab(capture_area)
                yield _mss_to_bgr(shot), time.time() - start
    else:
        raise ValueError(f"unknown capture_source: {capture_source}")


def _open_video_writer(output: Path, fps: int, sample_frame, codec: str = "mp4v"):
    height, width = sample_frame.shape[:2]
    return cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))


def capture_screen_image(output: Path, monitor: int = 1, region: dict | None = None, mss_module=None) -> Path:
    mss = mss_module or _load_mss()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        monitors = sct.monitors
        index = monitor if monitor < len(monitors) else len(monitors) - 1
        sct.shot(mon=index, output=str(output))
    return output


def _mss_to_bgr(shot) -> object:
    import numpy as np

    height = int(getattr(shot, "height", getattr(shot, "size", (0, 0))[1]))
    width = int(getattr(shot, "width", getattr(shot, "size", (0, 0))[0]))
    array = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


DETECTION_CSV_COLUMNS = ["frame_idx", "timestamp_seconds", "class_name", "confidence", "x1", "y1", "x2", "y2"]


def _write_detections_csv(detections: list[dict], output: Path) -> Path:
    import pandas as pd

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(detections, columns=DETECTION_CSV_COLUMNS)
    frame.to_csv(output, index=False)
    return output


def _draw_boxes(frame, rows: list[dict]):
    annotated = frame.copy()
    for row in rows:
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{row['class_name']} {float(row['confidence']):.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def capture_and_detect(
    capture_source: str,
    *,
    model_path=None,
    runner=None,
    duration_seconds: float | None = None,
    fps: int = 20,
    monitor: int = 1,
    region: dict | None = None,
    device: int = 0,
    output_video: Path | None = None,
    detections_output: Path,
    annotate: bool = False,
    codec: str = "mp4v",
    mss_module=None,
) -> dict[str, Path]:
    """Capture a video source while running live object detection on each frame.

    Writes a detections table (``frame_idx``, ``timestamp_seconds``, ``class_name``,
    ``confidence``, box coordinates) and, optionally, an annotated video. The
    footage is therefore processed as it is captured, not only after saving.
    """

    from soccer_edge.object_model import LocalObjectRunner

    if runner is None:
        if model_path is None:
            raise ValueError("model_path or runner is required for detection")
        runner = LocalObjectRunner(model_path)

    detections: list[dict] = []
    writer = None
    frame_idx = 0
    frames_written = 0
    try:
        for frame, elapsed in _iter_frames(
            capture_source,
            duration_seconds=duration_seconds if duration_seconds is not None else DEFAULT_DURATION_SECONDS,
            fps=fps,
            monitor=monitor,
            region=region,
            device=device,
            mss_module=mss_module,
        ):
            rows = runner(frame)
            for row in rows:
                detections.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_seconds": round(elapsed, 3),
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                        "x1": float(row["x1"]),
                        "y1": float(row["y1"]),
                        "x2": float(row["x2"]),
                        "y2": float(row["y2"]),
                    }
                )
            display = _draw_boxes(frame, rows) if annotate else frame
            if output_video is not None:
                if writer is None:
                    writer = _open_video_writer(output_video, fps, display, codec)
                writer.write(display)
                frames_written += 1
            frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
    detections_path = _write_detections_csv(detections, detections_output)
    saved_video = Path(output_video) if (output_video is not None and frames_written > 0) else None
    return {"video": saved_video, "detections": detections_path}


def capture_and_detect_and_register(
    capture_source: str,
    output: Path,
    *,
    model_path=None,
    runner=None,
    duration_seconds: float | None = None,
    fps: int = 20,
    monitor: int = 1,
    region: dict | None = None,
    device: int = 0,
    detections_output: Path,
    annotate: bool = False,
    manifest_path: Path = Path("manifests/video_manifest.csv"),
    rights_status: str,
    rights_reference: str,
    video_id: str | None = None,
    clip_type: str = "training_clip",
    match_id: str = "",
    competition: str = "",
    season: str = "",
    home_team: str = "",
    away_team: str = "",
    period: str = "",
    start_match_second: float | None = None,
    end_match_second: float | None = None,
    notes: str = "",
    codec: str = "mp4v",
    mss_module=None,
) -> tuple[Path, Path, VideoManifestRow]:
    if capture_source == "image":
        raise ValueError("live detection requires a video source (screen/webcam), not image")
    result = capture_and_detect(
        capture_source,
        model_path=model_path,
        runner=runner,
        duration_seconds=duration_seconds,
        fps=fps,
        monitor=monitor,
        region=region,
        device=device,
        output_video=output,
        detections_output=detections_output,
        annotate=annotate,
        codec=codec,
        mss_module=mss_module,
    )
    saved_video = result["video"]
    detections_path = result["detections"]
    if saved_video is None:
        raise RuntimeError(
            "capture produced no frames; no video written and manifest not updated "
            f"(detections table at {detections_path})"
        )
    resolved_video_id = video_id or f"capture-{_utc_stamp()}"
    row = build_capture_row(
        video_id=resolved_video_id,
        local_path=saved_video,
        rights_status=rights_status,
        rights_reference=rights_reference,
        capture_source=capture_source,
        clip_type=clip_type,
        match_id=match_id,
        competition=competition,
        season=season,
        home_team=home_team,
        away_team=away_team,
        period=period,
        start_match_second=start_match_second,
        end_match_second=end_match_second,
        notes=notes,
    )
    from soccer_edge.video.manifest import append_manifest_row

    append_manifest_row(manifest_path, row)
    return saved_video, detections_path, row


def build_capture_row(
    video_id: str,
    local_path: Path,
    rights_status: str,
    rights_reference: str,
    capture_source: str,
    clip_type: str = "training_clip",
    match_id: str = "",
    competition: str = "",
    season: str = "",
    home_team: str = "",
    away_team: str = "",
    period: str = "",
    start_match_second: float | None = None,
    end_match_second: float | None = None,
    notes: str = "",
) -> VideoManifestRow:
    _require_rights(rights_status, rights_reference)
    return VideoManifestRow(
        video_id=video_id,
        match_id=match_id,
        competition=competition,
        season=season,
        home_team=home_team,
        away_team=away_team,
        clip_type=clip_type,
        source_url=f"{CAPTURE_SCHEME}://{capture_source}",
        local_path=Path(local_path),
        period=period,
        start_match_second=start_match_second,
        end_match_second=end_match_second,
        rights_status=rights_status,
        rights_reference=rights_reference,
        notes=notes,
    )


def capture_and_register(
    capture_source: str,
    output: Path,
    *,
    duration_seconds: float | None = None,
    fps: int = 20,
    monitor: int = 1,
    region: dict | None = None,
    device: int = 0,
    manifest_path: Path = Path("manifests/video_manifest.csv"),
    rights_status: str,
    rights_reference: str,
    video_id: str | None = None,
    clip_type: str = "training_clip",
    match_id: str = "",
    competition: str = "",
    season: str = "",
    home_team: str = "",
    away_team: str = "",
    period: str = "",
    start_match_second: float | None = None,
    end_match_second: float | None = None,
    notes: str = "",
    mss_module=None,
) -> tuple[Path, VideoManifestRow]:
    if capture_source == "webcam":
        saved = capture_webcam_video(
            output,
            duration_seconds=duration_seconds if duration_seconds is not None else DEFAULT_DURATION_SECONDS,
            fps=fps,
            device=device,
        )
    elif capture_source == "screen":
        if duration_seconds is None:
            raise ValueError("duration_seconds is required for screen video capture")
        saved = capture_screen_video(
            output, duration_seconds, fps=fps, monitor=monitor, region=region, mss_module=mss_module
        )
    elif capture_source == "image":
        saved = capture_screen_image(output, monitor=monitor, region=region, mss_module=mss_module)
    else:
        raise ValueError(f"unknown capture_source: {capture_source}")

    resolved_video_id = video_id or f"capture-{_utc_stamp()}"
    row = build_capture_row(
        video_id=resolved_video_id,
        local_path=saved,
        rights_status=rights_status,
        rights_reference=rights_reference,
        capture_source=capture_source,
        clip_type=clip_type,
        match_id=match_id,
        competition=competition,
        season=season,
        home_team=home_team,
        away_team=away_team,
        period=period,
        start_match_second=start_match_second,
        end_match_second=end_match_second,
        notes=notes,
    )
    from soccer_edge.video.manifest import append_manifest_row

    append_manifest_row(manifest_path, row)
    return saved, row
