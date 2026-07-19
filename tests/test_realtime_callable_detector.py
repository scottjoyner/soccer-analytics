from soccer_edge.realtime.realtime_state import RealtimeDetector


class CallableDetector:
    def __call__(self, frame):
        return [
            {
                "class_name": "player",
                "confidence": 0.9,
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 20,
            }
        ]


def test_realtime_detector_accepts_callable_runner() -> None:
    detector = RealtimeDetector(CallableDetector())
    states = detector.process_frame(object(), timestamp_seconds=0.0)
    assert len(states) == 1
    assert states[0].class_name == "player"
