
import pytest

from soccer_edge.video.yolo_pipeline import run_yolo_detection


def test_run_yolo_detection_refuses_ungated_footage(tmp_path) -> None:
    # The rights gate is on by default: no manifest row => refuse, before any file opens.
    with pytest.raises(ValueError, match="requires an approved manifest row"):
        run_yolo_detection(
            input_path=tmp_path / "clip.mp4",
            output_dir=tmp_path / "out",
            model_path="yolov8n.pt",
        )


def test_run_yolo_detection_refuses_missing_manifest_row(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,match_id,clip_type,local_path,rights_status,rights_reference\n"
        "ok,match_1,full_match,clip.mp4,licensed,license-file:///perms/x.pdf\n",
        encoding="utf-8",
    )
    # rights_video_id points at a row whose local_path does not match input => gate fails.
    with pytest.raises(ValueError):
        run_yolo_detection(
            input_path=tmp_path / "other.mp4",
            output_dir=tmp_path / "out",
            model_path="yolov8n.pt",
            rights_manifest=manifest,
            rights_video_id="ok",
            licensed_root=tmp_path,
        )


def test_run_yolo_detection_allows_explicit_bypass_flag(tmp_path) -> None:
    # Callers processing synthetic/pre-approved frames may opt out; the gate no-ops and
    # the function proceeds (it will still fail later if the model/path is invalid, but
    # the rights gate is the thing under test here).
    try:
        run_yolo_detection(
            input_path=tmp_path / "clip.mp4",
            output_dir=tmp_path / "out",
            model_path="yolov8n.pt",
            enforce_rights=False,
        )
    except ValueError as exc:
        # Expected only if/when the model or media reader fails; the gate must NOT be the cause.
        assert "requires an approved manifest row" not in str(exc)
