from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .models import BBox, Pose, StagObservation, bbox_area, bbox_from_points, bbox_iou, clip_bbox


@dataclass(slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    @classmethod
    def from_npz(cls, path: str) -> "CameraCalibration":
        data = np.load(path)
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
        )


class StagDetector:
    def __init__(
        self,
        library_hd: int = 17,
        marker_size: Optional[float] = None,
        calibration: Optional[CameraCalibration] = None,
        roi_padding: int = 12,
    ) -> None:
        try:
            import stag
        except ImportError as exc:
            raise RuntimeError(
                "stag-python is not installed or no `stag` module is available. "
                "Install a compatible STag Python binding and verify `import stag`."
            ) from exc

        self._stag = stag
        self.library_hd = library_hd
        self.marker_size = marker_size
        self.calibration = calibration
        self.roi_padding = roi_padding

    def detect(
        self,
        frame: np.ndarray,
        rois: Optional[list[BBox]] = None,
    ) -> list[StagObservation]:
        height, width = frame.shape[:2]
        if rois is None or not rois:
            rois = [(0, 0, width, height)]

        observations: list[StagObservation] = []
        seen: set[tuple[int, int, int]] = set()

        for roi in rois:
            x, y, w, h = clip_bbox(roi, width, height, self.roi_padding)
            if w <= 4 or h <= 4:
                continue

            crop = frame[y : y + h, x : x + w]
            for obs in self._detect_crop(crop, x, y):
                key = (obs.marker_id, int(obs.bbox[0]), int(obs.bbox[1]))
                if key in seen:
                    continue
                seen.add(key)
                observations.append(obs)

        return dedupe_observations(observations)

    def _detect_crop(
        self,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[StagObservation]:
        corners, ids = self._call_stag(crop)
        if ids is None or corners is None:
            return []

        observations: list[StagObservation] = []
        flat_ids = np.asarray(ids).reshape(-1)
        for raw_corners, marker_id in zip(corners, flat_ids):
            marker_corners = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 2)
            if marker_corners.shape[0] != 4:
                continue

            marker_corners[:, 0] += offset_x
            marker_corners[:, 1] += offset_y
            bbox = bbox_from_points(marker_corners)
            pose = self._estimate_pose(marker_corners)
            observations.append(
                StagObservation(
                    marker_id=int(marker_id),
                    corners=marker_corners,
                    bbox=bbox,
                    pose=pose,
                )
            )
        return observations

    def _call_stag(self, image: np.ndarray) -> tuple[object, object]:
        try:
            corners, ids, _ = self._stag.detectMarkers(
                image,
                libraryHD=self.library_hd,
            )
            return corners, ids
        except TypeError:
            corners, ids, _ = self._stag.detectMarkers(image, self.library_hd)
            return corners, ids

    def _estimate_pose(self, corners: np.ndarray) -> Optional[Pose]:
        if self.marker_size is None or self.calibration is None:
            return None

        half = self.marker_size / 2.0
        object_points = np.asarray(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            np.asarray(corners, dtype=np.float32),
            self.calibration.camera_matrix,
            self.calibration.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None
        return Pose(rvec=rvec, tvec=tvec)


def dedupe_observations(
    observations: list[StagObservation],
    iou_threshold: float = 0.1,
) -> list[StagObservation]:
    deduped: list[StagObservation] = []
    for observation in sorted(
        observations,
        key=lambda obs: bbox_area(obs.bbox),
        reverse=True,
    ):
        duplicate = False
        for kept in deduped:
            if kept.marker_id != observation.marker_id:
                continue
            if bbox_iou(kept.bbox, observation.bbox) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            deduped.append(observation)
    return deduped
