from soccer_edge.video.detector import Detection, NullDetector
from soccer_edge.video.tracker import FrameLocalTracker


def test_detection_center() -> None:
    detection = Detection(frame_idx=1, class_name="ball", confidence=0.9, x1=0, y1=0, x2=10, y2=20)
    assert detection.center == (5.0, 10.0)


def test_null_detector_returns_empty_list() -> None:
    detector = NullDetector()
    assert detector.detect_frame(0, object()) == []


def test_frame_local_tracker_assigns_ids() -> None:
    tracker = FrameLocalTracker()
    detections = [Detection(frame_idx=7, class_name="player", confidence=0.8, x1=1, y1=2, x2=3, y2=4)]
    states = tracker.update(detections)
    assert states[0].track_id == "7:0"
    assert states[0].class_name == "player"
