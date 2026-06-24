from dataclasses import dataclass

from soccer_edge.video.detector import Detection


@dataclass(frozen=True)
class TrackState:
    frame_idx: int
    track_id: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class FrameLocalTracker:
    def update(self, detections: list[Detection]) -> list[TrackState]:
        states: list[TrackState] = []
        for idx, detection in enumerate(detections):
            states.append(
                TrackState(
                    frame_idx=detection.frame_idx,
                    track_id=f"{detection.frame_idx}:{idx}",
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                )
            )
        return states
