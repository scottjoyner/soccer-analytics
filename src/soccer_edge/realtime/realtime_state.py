"""Frame-by-frame realtime state management for live YOLO detection.

A thin stateful layer over a per-frame detector that the offline pipeline lacks:

* **Rolling buffer** - keeps only the last ``max_buffer_frames`` frames in memory
  so a long live session does not grow unbounded.
* **FPS throttle** - when the capture source runs faster than the model can keep up,
  only sample one frame per ``min_interval_seconds`` and skip the rest.
* **Track / ball continuity** - the offline ``FrameLocalTracker`` renumbers tracks
  every frame; here we carry a persistent ``track_id`` across frames using a tiny
  nearest-centroid matcher, so a player/ball keeps one identity over time. This is
  what lets us build trajectories and a ball-state series.
* **Low-confidence review queue** - detections below ``review_threshold`` but above
  the model confidence floor are queued for human review (the same path the
  offline pipeline routes to ``video sample-low-confidence``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from soccer_edge.video.detector import Detection
from soccer_edge.video.tracker import TrackState


@dataclass(frozen=True)
class ReviewQueueEntry:
    frame_idx: int
    timestamp_seconds: float
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class RealtimeState:
    """Mutable per-session realtime detection state.

    Holds the rolling frame buffer, the latest matched tracks, the ball-state series,
    and the low-confidence review queue. Built to be fed one ``Detection`` list per
    frame via :meth:`ingest`.
    """

    video_width: float = 1920.0
    video_height: float = 1080.0
    max_buffer_frames: int = 300
    min_interval_seconds: float = 0.0
    review_threshold: float = 0.4
    confidence_floor: float = 0.25

    buffer: list[list[Detection]] = field(default_factory=list)
    track_memory: dict[str, TrackState] = field(default_factory=dict)
    ball_series: list[dict] = field(default_factory=list)
    review_queue: list[ReviewQueueEntry] = field(default_factory=list)
    _last_ingest_t: float = -1.0
    _next_track_id: int = 0

    def _throttle(self, timestamp_seconds: float) -> bool:
        if self.min_interval_seconds <= 0.0:
            return False
        if self._last_ingest_t < 0:
            return False
        return (timestamp_seconds - self._last_ingest_t) < self.min_interval_seconds

    def _match_tracks(self, detections: list[Detection]) -> list[TrackState]:
        """Carry track identity across frames via class-aware nearest-centroid matching.

        Each detection this frame is matched to the last-seen track of the SAME
        class by closest box center; unmatched detections start a fresh track id.
        Restricting to the same class keeps players and the ball from swapping
        identities. This is a deliberately simple, calibration-free matcher
        (no ReID, no Kalman) - enough to give the ball-state series and
        trajectories stable identities for live state.
        """

        prev = list(self.track_memory.values())
        states: list[TrackState] = []
        used_prev: set[int] = set()
        for detection in detections:
            best_idx = -1
            best_dist = float("inf")
            dc = detection.center
            for idx, p in enumerate(prev):
                if idx in used_prev:
                    continue
                if p.class_name != detection.class_name:
                    continue
                pc = ((p.x1 + p.x2) / 2.0, (p.y1 + p.y2) / 2.0)
                dist = (dc[0] - pc[0]) ** 2 + (dc[1] - pc[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0 and best_dist <= (max(self.video_width, self.video_height) * 0.15) ** 2:
                track_id = prev[best_idx].track_id
                used_prev.add(best_idx)
            else:
                track_id = f"t{self._next_track_id}"
                self._next_track_id += 1
            state = TrackState(
                frame_idx=detection.frame_idx,
                track_id=track_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
            )
            states.append(state)
        # Remember only the tracks seen this frame.
        self.track_memory = {s.track_id: s for s in states}
        return states

    def ingest(
        self,
        frame_idx: int,
        timestamp_seconds: float | None,
        detections: list[Detection],
    ) -> list[TrackState]:
        """Ingest one frame's detections; returns the matched track states.

        Skipped frames (throttled) return an empty list and change nothing.
        """

        if timestamp_seconds is not None and self._throttle(timestamp_seconds):
            return []
        self._last_ingest_t = timestamp_seconds if timestamp_seconds is not None else self._last_ingest_t

        kept: list[Detection] = []
        for detection in detections:
            if detection.confidence < self.confidence_floor:
                continue
            kept.append(detection)
            # Queue detections that clear the model floor but fall below the
            # human-review threshold: too unsure to auto-accept, too confident
            # to discard.
            if self.confidence_floor <= detection.confidence < self.review_threshold:
                self.review_queue.append(
                    ReviewQueueEntry(
                        frame_idx=frame_idx,
                        timestamp_seconds=timestamp_seconds if timestamp_seconds is not None else 0.0,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        x1=detection.x1,
                        y1=detection.y1,
                        x2=detection.x2,
                        y2=detection.y2,
                    )
                )
        self.buffer.append(kept)
        if len(self.buffer) > self.max_buffer_frames:
            self.buffer.pop(0)

        states = self._match_tracks(kept)
        for state in states:
            if str(state.class_name).lower() in {"ball", "sports ball"}:
                self.ball_series.append(
                    {
                        "frame_idx": state.frame_idx,
                        "timestamp_seconds": timestamp_seconds if timestamp_seconds is not None else 0.0,
                        "x_center": (state.x1 + state.x2) / 2.0,
                        "y_center": (state.y1 + state.y2) / 2.0,
                        "confidence": state.confidence,
                    }
                )
        return states

    def recent_detections(self, window_frames: int | None = None) -> list[Detection]:
        """Flatten the rolling buffer (optionally the last ``window_frames``)."""

        frames = self.buffer if window_frames is None else self.buffer[-window_frames:]
        out: list[Detection] = []
        for frame in frames:
            out.extend(frame)
        return out

    def frame_rows(self) -> list[dict]:
        """Flat detection rows (frame_idx, class_name, box, ...) for the buffer."""

        rows: list[dict] = []
        for frame in self.buffer:
            for detection in frame:
                rows.append(
                    {
                        "frame_idx": detection.frame_idx,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                    }
                )
        return rows

    def write_state_tables(self, output_dir: Path, video_id: str = "") -> dict[str, Path]:
        """Persist the buffer + tracks + ball series + review queue (parquet)."""

        from soccer_edge.video.state_tables import write_video_state_tables

        import pandas as pd

        track_rows = list(self.track_memory.values())
        ball_rows = [
            {
                "video_id": video_id,
                "frame_idx": b["frame_idx"],
                "timestamp_seconds": b["timestamp_seconds"],
                "x_m": b["x_center"],
                "y_m": b["y_center"],
                "confidence": b["confidence"],
                "interpolated": False,
            }
            for b in self.ball_series
        ]
        player_rows = [
            {
                "video_id": video_id,
                "frame_idx": s.frame_idx,
                "timestamp_seconds": 0.0,
                "track_id": s.track_id,
                "team": "",
                "x_m": (s.x1 + s.x2) / 2.0,
                "y_m": (s.y1 + s.y2) / 2.0,
                "confidence": s.confidence,
            }
            for s in track_rows
            if str(s.class_name).lower() in {"player", "person"}
        ]
        review_rows = [
            {
                "frame_idx": r.frame_idx,
                "timestamp_seconds": r.timestamp_seconds,
                "class_name": r.class_name,
                "confidence": r.confidence,
                "x1": r.x1,
                "y1": r.y1,
                "x2": r.x2,
                "y2": r.y2,
            }
            for r in self.review_queue
        ]
        paths = write_video_state_tables(
            output_dir=output_dir,
            detections=self.frame_rows(),
            tracks=track_rows,
            ball_states=ball_rows,
            player_states=player_rows,
        )
        review_path = output_dir / "review_queue.parquet"
        pd.DataFrame(review_rows).to_parquet(review_path, index=False)
        paths["review_queue"] = review_path
        return paths


class RealtimeDetector:
    """Wrap a per-frame detector with the realtime state manager.

    Feed frames via :meth:`process_frame`; the detector runs, detections are matched
    into tracks, and every ``window_seconds`` the accumulated window is handed to a
    callback (``on_window``) which typically builds the live match state + triggers.
    """

    def __init__(
        self,
        detector,
        *,
        video_width: float = 1920.0,
        video_height: float = 1080.0,
        max_buffer_frames: int = 300,
        min_interval_seconds: float = 0.0,
        review_threshold: float = 0.4,
        confidence_floor: float = 0.25,
        window_seconds: float = 10.0,
        on_window=None,
    ) -> None:
        self.detector = detector
        self.state = RealtimeState(
            video_width=video_width,
            video_height=video_height,
            max_buffer_frames=max_buffer_frames,
            min_interval_seconds=min_interval_seconds,
            review_threshold=review_threshold,
            confidence_floor=confidence_floor,
        )
        self.window_seconds = window_seconds
        self.on_window = on_window
        self._window_start: float | None = None
        self._frame_idx = 0

    def process_frame(self, frame, timestamp_seconds: float | None = None) -> list[TrackState]:
        raw = self.detector.detect_frame(self._frame_idx, frame)
        # Accept either Detection objects (YOLODetector) or raw dict rows
        # (LocalObjectRunner); convert the latter so the state manager always
        # receives Detection instances.
        if raw and not isinstance(raw[0], Detection):
            from soccer_edge.video.detector import detections_from_rows

            detections = detections_from_rows(raw, self._frame_idx, confidence_threshold=self.state.confidence_floor)
        else:
            detections = raw
        t = timestamp_seconds if timestamp_seconds is not None else float(self._frame_idx)
        states = self.state.ingest(self._frame_idx, t, detections)
        self._frame_idx += 1
        if self._window_start is None:
            self._window_start = t
        elif t - self._window_start >= self.window_seconds:
            if self.on_window is not None:
                self.on_window(self.state, self._window_start, t)
            self._window_start = t
        return states

    def flush(self) -> None:
        if self.on_window is not None and self._window_start is not None:
            self.on_window(self.state, self._window_start, float(self._frame_idx))
