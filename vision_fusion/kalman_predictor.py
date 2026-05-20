from __future__ import annotations

import cv2
import numpy as np

from .models import Track, bbox_center


class KalmanPredictor:
    """Per-track Kalman filter for lightweight inter-frame prediction.

    Replaces optical flow when async detection provides corrections.
    Prediction cost: <0.1ms per track vs ~8ms for optical flow.
    """

    def __init__(
        self,
        process_noise: float = 4.0,
        measurement_noise: float = 1.0,
    ) -> None:
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._filters: dict[int, cv2.KalmanFilter] = {}
        self._corner_offsets: dict[int, np.ndarray] = {}

    def _create_filter(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(8, 4, 0)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        kf.transitionMatrix[0, 4] = 1.0
        kf.transitionMatrix[1, 5] = 1.0
        kf.transitionMatrix[2, 6] = 1.0
        kf.transitionMatrix[3, 7] = 1.0

        kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0
        kf.measurementMatrix[2, 2] = 1.0
        kf.measurementMatrix[3, 3] = 1.0

        kf.processNoiseCov = np.eye(8, dtype=np.float32) * self._process_noise
        kf.processNoiseCov[4:, 4:] *= 2.0

        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * self._measurement_noise

        kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        return kf

    def init_track(self, track: Track) -> None:
        kf = self._create_filter()
        cx, cy = bbox_center(track.bbox)
        x, y, w, h = track.bbox
        state = np.array(
            [cx, cy, float(w), float(h), 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        kf.statePost = state.reshape(8, 1)
        self._filters[track.track_id] = kf

        if track.corners is not None:
            corners = np.asarray(track.corners, dtype=np.float32).reshape(-1, 2)
            self._corner_offsets[track.track_id] = corners - np.array(
                [[cx, cy]], dtype=np.float32
            )

    def predict_track(self, track: Track) -> None:
        kf = self._filters.get(track.track_id)
        if kf is None:
            self.init_track(track)
            return

        prediction = kf.predict().flatten()
        cx = float(prediction[0])
        cy = float(prediction[1])
        w = max(1.0, float(prediction[2]))
        h = max(1.0, float(prediction[3]))
        vx = float(prediction[4])
        vy = float(prediction[5])

        track.bbox = (
            int(round(cx - w / 2)),
            int(round(cy - h / 2)),
            int(round(w)),
            int(round(h)),
        )
        track.velocity = (vx, vy)
        track.source = "kalman"
        track.age += 1
        track.missed += 1

        if track.corners is not None and track.track_id in self._corner_offsets:
            offsets = self._corner_offsets[track.track_id]
            center = np.array([[cx, cy]], dtype=np.float32)
            track.corners = (offsets + center).reshape(-1, 2)

    def correct_track(self, track: Track) -> None:
        kf = self._filters.get(track.track_id)
        if kf is None:
            self.init_track(track)
            return

        cx, cy = bbox_center(track.bbox)
        _, _, w, h = track.bbox
        measurement = np.array(
            [cx, cy, float(w), float(h)], dtype=np.float32
        ).reshape(4, 1)
        kf.correct(measurement)

        if track.corners is not None:
            corners = np.asarray(track.corners, dtype=np.float32).reshape(-1, 2)
            self._corner_offsets[track.track_id] = corners - np.array(
                [[cx, cy]], dtype=np.float32
            )

    def remove_track(self, track_id: int) -> None:
        self._filters.pop(track_id, None)
        self._corner_offsets.pop(track_id, None)
