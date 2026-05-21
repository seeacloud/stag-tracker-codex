from __future__ import annotations

from typing import Optional

import numpy as np

from .kalman_predictor import KalmanPredictor
from .models import (
    BBox,
    Detection,
    StagObservation,
    Track,
    append_track_history,
    bbox_area,
    bbox_center,
    bbox_iou,
)
from .one_euro_filter import OneEuroFilter2D
from .optical_flow import OpticalFlowTracker


class FusionTracker:
    def __init__(
        self,
        flow: Optional[OpticalFlowTracker] = None,
        kalman: Optional[KalmanPredictor] = None,
        predictor: str = "flow",
        iou_threshold: float = 0.25,
        max_missed: int = 20,
        smooth_alpha: float = 0.35,
        smooth_deadband: float = 6.0,
        smooth_snap: float = 70.0,
        oef_min_cutoff: float = 1.0,
        oef_beta: float = 0.007,
        oef_dcutoff: float = 1.0,
        oef_fps: float = 60.0,
    ) -> None:
        self.predictor_mode = predictor
        self.flow = flow
        self.kalman = kalman or (KalmanPredictor() if predictor == "kalman" else None)
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.smooth_deadband = smooth_deadband
        self.smooth_snap = smooth_snap
        # 1-Euro Filter parameters (per-track filters stored in _oef_filters)
        self._oef_min_cutoff = oef_min_cutoff
        self._oef_beta = oef_beta
        self._oef_dcutoff = oef_dcutoff
        self._oef_fps = oef_fps
        # Per-track filter state: track_id -> {"bbox": OneEuroFilter2D, "corners": OneEuroFilter2D}
        self._oef_filters: dict[int, dict[str, OneEuroFilter2D]] = {}
        self.tracks: list[Track] = []
        self._next_track_id = 1

    def predict(
        self,
        previous_gray: Optional[np.ndarray] = None,
        gray: Optional[np.ndarray] = None,
    ) -> list[Track]:
        if self.predictor_mode == "kalman" and self.kalman is not None:
            for track in self.tracks:
                self.kalman.predict_track(track)
        elif self.flow is not None and previous_gray is not None and gray is not None:
            self.tracks = [
                self.flow.update_track(previous_gray, gray, track)
                for track in self.tracks
            ]
        for track in self.tracks:
            self._stabilize_track(track)
        self._merge_duplicate_marker_tracks()
        self._drop_stale()
        return self.tracks

    def record_predicted_history(self) -> list[Track]:
        for track in self.tracks:
            append_track_history(track)
        return self.tracks

    def update(
        self,
        gray: np.ndarray,
        detections: list[Detection],
        observations: list[StagObservation],
    ) -> list[Track]:
        observations = self._dedupe_observations(observations)
        measurements = self._build_measurements(detections, observations)
        matched_track_ids: set[int] = set()

        for measurement in measurements:
            track = self._match_measurement(measurement, matched_track_ids)
            if track is None:
                track = self._new_track(gray, measurement)
                self.tracks.append(track)
            else:
                self._apply_measurement(gray, track, measurement)
            matched_track_ids.add(track.track_id)

        self._merge_duplicate_marker_tracks()
        self._drop_stale()
        self._record_unmatched_history(matched_track_ids)
        return self.tracks

    def _dedupe_observations(
        self,
        observations: list[StagObservation],
    ) -> list[StagObservation]:
        by_marker: dict[int, list[StagObservation]] = {}
        markerless: list[StagObservation] = []
        for observation in observations:
            if observation.marker_id is None:
                markerless.append(observation)
            else:
                by_marker.setdefault(observation.marker_id, []).append(observation)

        deduped = markerless[:]
        for marker_id, group in by_marker.items():
            if len(group) == 1:
                deduped.append(group[0])
                continue
            deduped.append(max(group, key=lambda obs: self._observation_score(marker_id, obs)))
        return deduped

    def _observation_score(self, marker_id: int, observation: StagObservation) -> float:
        matching_tracks = [track for track in self.tracks if track.marker_id == marker_id]
        if matching_tracks:
            best_iou = max(bbox_iou(track.bbox, observation.bbox) for track in matching_tracks)
            return best_iou * 100000.0 + bbox_area(observation.bbox)
        return float(bbox_area(observation.bbox))

    def _build_measurements(
        self,
        detections: list[Detection],
        observations: list[StagObservation],
    ) -> list[dict[str, object]]:
        measurements: list[dict[str, object]] = []

        for obs in observations:
            parent = self._find_parent_detection(obs, detections)
            label = f"stag:{obs.marker_id}"
            confidence = 1.0
            if parent is not None:
                label = f"{parent.label} stag:{obs.marker_id}"
                confidence = max(confidence, parent.confidence)
            measurements.append(
                {
                    "bbox": obs.bbox,
                    "label": label,
                    "confidence": confidence,
                    "marker_id": obs.marker_id,
                    "corners": obs.corners,
                    "pose": obs.pose,
                    "source": "stag",
                }
            )

        for det in detections:
            if any(self._contains_observation(det, obs) for obs in observations):
                continue
            measurements.append(
                {
                    "bbox": det.bbox,
                    "label": det.label,
                    "confidence": det.confidence,
                    "marker_id": None,
                    "corners": None,
                    "pose": None,
                    "source": "yolo",
                }
            )

        return measurements

    @staticmethod
    def _contains_observation(detection: Detection, observation: StagObservation) -> bool:
        cx, cy = bbox_center(observation.bbox)
        x, y, w, h = detection.bbox
        return x <= cx <= x + w and y <= cy <= y + h

    @staticmethod
    def _find_parent_detection(
        observation: StagObservation,
        detections: list[Detection],
    ) -> Detection | None:
        cx, cy = bbox_center(observation.bbox)
        for detection in detections:
            x, y, w, h = detection.bbox
            if x <= cx <= x + w and y <= cy <= y + h:
                return detection
        return None

    def _match_measurement(
        self,
        measurement: dict[str, object],
        used_track_ids: set[int],
    ) -> Track | None:
        marker_id = measurement["marker_id"]
        if marker_id is not None:
            candidates = [
                track
                for track in self.tracks
                if track.track_id not in used_track_ids and track.marker_id == marker_id
            ]
            if candidates:
                bbox = measurement["bbox"]
                return max(
                    candidates,
                    key=lambda track: (
                        bbox_iou(track.bbox, bbox),  # type: ignore[arg-type]
                        -track.detection_missed,
                        -track.missed,
                        -track.track_id,
                    ),
                )

        bbox = measurement["bbox"]
        best_track: Track | None = None
        best_iou = self.iou_threshold
        for track in self.tracks:
            if track.track_id in used_track_ids:
                continue
            score = bbox_iou(track.bbox, bbox)  # type: ignore[arg-type]
            if score > best_iou:
                best_iou = score
                best_track = track
        return best_track

    def _new_track(self, gray: np.ndarray, measurement: dict[str, object]) -> Track:
        bbox = measurement["bbox"]
        if self.predictor_mode == "kalman":
            points = np.empty((0, 1, 2), dtype=np.float32)
        elif self.flow is not None:
            points = self.flow.seed_points(gray, bbox)  # type: ignore[arg-type]
        else:
            points = np.empty((0, 1, 2), dtype=np.float32)
        track = Track(
            track_id=self._next_track_id,
            bbox=bbox,  # type: ignore[arg-type]
            points=points,
            label=str(measurement["label"]),
            confidence=float(measurement["confidence"]),
            marker_id=measurement["marker_id"],  # type: ignore[arg-type]
            corners=measurement["corners"],  # type: ignore[arg-type]
            pose=measurement["pose"],  # type: ignore[arg-type]
            source=str(measurement["source"]),
        )
        self._stabilize_track(track, reset=True)
        append_track_history(track)
        self._next_track_id += 1
        if self.predictor_mode == "kalman" and self.kalman is not None:
            self.kalman.init_track(track)
        return track

    def _apply_measurement(
        self,
        gray: np.ndarray,
        track: Track,
        measurement: dict[str, object],
    ) -> None:
        track.bbox = measurement["bbox"]  # type: ignore[assignment]
        label = str(measurement["label"])
        if measurement["marker_id"] is None and track.marker_id is not None:
            label = f"{label} stag:{track.marker_id}"
        track.label = label
        track.confidence = float(measurement["confidence"])
        if measurement["marker_id"] is not None:
            track.marker_id = measurement["marker_id"]  # type: ignore[assignment]
            track.corners = measurement["corners"]  # type: ignore[assignment]
            track.pose = measurement["pose"]  # type: ignore[assignment]
        track.missed = 0
        track.detection_missed = 0
        track.source = str(measurement["source"])
        if self.predictor_mode == "kalman" and self.kalman is not None:
            self.kalman.correct_track(track)
        elif self.flow is not None:
            self.flow.refresh_points(gray, track)
        self._stabilize_track(track)
        append_track_history(track)

    def _drop_stale(self) -> None:
        kept: list[Track] = []
        for track in self.tracks:
            if max(track.missed, track.detection_missed) <= self.max_missed:
                kept.append(track)
            else:
                # Clean up filter state for dropped tracks
                self._oef_filters.pop(track.track_id, None)
                if self.predictor_mode == "kalman" and self.kalman is not None:
                    self.kalman.remove_track(track.track_id)
        self.tracks = kept

    def _record_unmatched_history(self, matched_track_ids: set[int]) -> None:
        for track in self.tracks:
            if track.track_id not in matched_track_ids:
                track.detection_missed += 1
                append_track_history(track)

    def _merge_duplicate_marker_tracks(self) -> None:
        best_by_marker: dict[int, Track] = {}
        markerless: list[Track] = []

        for track in self.tracks:
            if track.marker_id is None:
                markerless.append(track)
                continue

            current = best_by_marker.get(track.marker_id)
            if current is None or self._track_rank(track) > self._track_rank(current):
                best_by_marker[track.marker_id] = track

        new_tracks = markerless + sorted(
            best_by_marker.values(),
            key=lambda track: track.track_id,
        )
        # Clean up filter state for merged-away tracks
        kept_ids = {t.track_id for t in new_tracks}
        for tid in list(self._oef_filters.keys()):
            if tid not in kept_ids:
                del self._oef_filters[tid]
        self.tracks = new_tracks

    @staticmethod
    def _track_rank(track: Track) -> tuple[int, int, int, int, int]:
        return (
            1 if track.source == "stag" else 0,
            -track.detection_missed,
            -track.missed,
            track.age,
            -track.track_id,
        )

    def _stabilize_track(self, track: Track, reset: bool = False) -> None:
        track.display_bbox = self._smooth_bbox(track.track_id, track.display_bbox, track.bbox, reset)
        if track.corners is None:
            track.display_corners = None
        else:
            track.display_corners = self._smooth_points(
                track.track_id,
                track.display_corners,
                track.corners,
                reset,
            )

    def _get_oef(self, track_id: int, key: str) -> OneEuroFilter2D:
        """Get or create a 1-Euro Filter for a specific track and signal."""
        filters = self._oef_filters.get(track_id)
        if filters is None:
            filters = {}
            self._oef_filters[track_id] = filters
        filt = filters.get(key)
        if filt is None:
            filt = OneEuroFilter2D(
                min_cutoff=self._oef_min_cutoff,
                beta=self._oef_beta,
                dcutoff=self._oef_dcutoff,
                fps=self._oef_fps,
            )
            filters[key] = filt
        return filt

    def _smooth_bbox(
        self,
        track_id: int,
        previous: tuple[float, float, float, float] | None,
        current: BBox,
        reset: bool,
    ) -> tuple[float, float, float, float]:
        target = np.asarray(current, dtype=np.float32)
        if reset or previous is None:
            filt = self._get_oef(track_id, "bbox")
            filt.reset(target.reshape(1, 4))
            return tuple(float(v) for v in target)

        prev = np.asarray(previous, dtype=np.float32)
        center_delta = np.linalg.norm(
            np.asarray(bbox_center(current), dtype=np.float32)
            - np.asarray(
                [
                    prev[0] + prev[2] / 2.0,
                    prev[1] + prev[3] / 2.0,
                ],
                dtype=np.float32,
            )
        )
        # Snap threshold: teleport bypasses filter
        if center_delta >= self.smooth_snap:
            filt = self._get_oef(track_id, "bbox")
            filt.reset(target.reshape(1, 4))
            return tuple(float(v) for v in target)

        # Apply 1-Euro Filter
        filt = self._get_oef(track_id, "bbox")
        smoothed = filt(target.reshape(1, 4))
        return tuple(float(v) for v in smoothed.flat)

    def _smooth_points(
        self,
        track_id: int,
        previous: np.ndarray | None,
        current: np.ndarray,
        reset: bool,
    ) -> np.ndarray:
        target = np.asarray(current, dtype=np.float32)
        if reset or previous is None or previous.shape != target.shape:
            filt = self._get_oef(track_id, "corners")
            filt.reset(target)
            return target.copy()

        # Check for teleport (snap threshold)
        deltas = np.linalg.norm(target.reshape(-1, 2) - previous.reshape(-1, 2), axis=1)
        mean_delta = float(np.mean(deltas))
        if mean_delta >= self.smooth_snap:
            filt = self._get_oef(track_id, "corners")
            filt.reset(target)
            return target.copy()

        # Apply 1-Euro Filter
        filt = self._get_oef(track_id, "corners")
        return filt(target)
