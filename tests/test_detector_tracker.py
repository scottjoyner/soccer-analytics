from soccer_edge.video.detector import Detection, NullDetector, detections_from_rows
from soccer_edge.video.tracker import FrameLocalTracker


def test_detection_center() -> None:
    detection = Detection(frame_idx=1, class_name="ball", confidence=0.9, x1=0, y1=0, x2=10, y2=20)
    assert detection.center == (5.0, 10.0)


def test_null_detector_returns_empty_list() -> None:
    detector = NullDetector()
    assert detector.detect_frame(0, object()) == []
    assert detector.detect(0, object()) == []


def test_detections_from_rows_filters_by_confidence() -> None:
    rows = [
        {"class_name": "player", "confidence": 0.9, "x1": 1, "y1": 2, "x2": 3, "y2": 4},
        {"class_name": "ball", "confidence": 0.1, "x1": 0, "y1": 0, "x2": 1, "y2": 1},
    ]
    detections = detections_from_rows(rows, frame_idx=4, confidence_threshold=0.25)
    assert len(detections) == 1
    assert detections[0].class_name == "player"
    assert detections[0].frame_idx == 4
    assert detections[0].x1 == 1.0



def test_frame_local_tracker_assigns_ids() -> None:
    tracker = FrameLocalTracker()
    detections = [Detection(frame_idx=7, class_name="player", confidence=0.8, x1=1, y1=2, x2=3, y2=4)]
    states = tracker.update(detections)
    assert states[0].track_id == "7:0"
    assert states[0].class_name == "player"
