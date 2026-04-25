from __future__ import annotations

import cv2
import numpy as np

from .models import BBox, Track, bbox_area, bbox_center, bbox_from_points, clip_bbox


class OpticalFlowTracker:
    def __init__(
        self,
        max_corners: int = 80,
        quality_level: float = 0.01,
        min_distance: int = 6,
        min_points: int = 8,
        use_affine: bool = True,
        ransac_reproj_threshold: float = 3.0,
        fb_max_error: float = 2.0,
        min_area_ratio: float = 0.65,
        max_area_ratio: float = 1.35,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_points = min_points
        self.use_affine = use_affine
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.fb_max_error = fb_max_error
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.lk_params = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }

    def seed_points(self, gray: np.ndarray, bbox: BBox) -> np.ndarray:
        height, width = gray.shape[:2]
        x, y, w, h = clip_bbox(bbox, width, height)
        if w <= 4 or h <= 4:
            return np.empty((0, 1, 2), dtype=np.float32)

        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32)

    def update_track(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        track: Track,
    ) -> Track:
        if track.points.size == 0:
            return self._miss(track)

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            track.points,
            None,
            **self.lk_params,
        )
        if next_points is None or status is None:
            return self._miss(track)

        good_old = track.points[status.reshape(-1) == 1].reshape(-1, 2)
        good_new = next_points[status.reshape(-1) == 1].reshape(-1, 2)
        good_old, good_new = self._filter_forward_backward(
            previous_gray,
            gray,
            good_old,
            good_new,
        )
        if len(good_new) < self.min_points:
            return self._miss(track)

        x, y, w, h = track.bbox
        height, width = gray.shape[:2]
        deltas = good_new - good_old
        dx, dy = np.median(deltas, axis=0)
        transform = self._estimate_transform(good_old, good_new)
        if transform is not None:
            bbox_points = np.asarray(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ],
                dtype=np.float32,
            )
            moved_bbox_points = self._transform_points(bbox_points, transform)
            new_bbox = clip_bbox(bbox_from_points(moved_bbox_points), width, height)
            if not self._valid_bbox_update(track.bbox, new_bbox):
                transform = None
                new_bbox = clip_bbox((x + int(round(dx)), y + int(round(dy)), w, h), width, height)
            elif track.corners is not None:
                track.corners = self._transform_points(track.corners, transform)

        if transform is None:
            new_bbox = clip_bbox((x + int(round(dx)), y + int(round(dy)), w, h), width, height)
            if track.corners is not None:
                track.corners = track.corners + np.asarray([dx, dy], dtype=np.float32)

        track.bbox = new_bbox
        track.points = good_new.reshape(-1, 1, 2).astype(np.float32)
        track.velocity = (float(dx), float(dy))
        if len(good_new) < max(self.min_points * 2, self.max_corners // 4):
            fresh_points = self.seed_points(gray, track.bbox)
            if len(fresh_points) > len(track.points):
                track.points = fresh_points
        track.missed = 0
        track.age += 1
        track.source = "flow"
        track.history.append(bbox_center(new_bbox))
        track.history = track.history[-64:]
        return track

    def _filter_forward_backward(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        old_points: np.ndarray,
        new_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fb_max_error <= 0 or len(new_points) == 0:
            return old_points, new_points

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            previous_gray,
            new_points.reshape(-1, 1, 2).astype(np.float32),
            None,
            **self.lk_params,
        )
        if back_points is None or back_status is None:
            return old_points, new_points

        back_points = back_points.reshape(-1, 2)
        errors = np.linalg.norm(back_points - old_points, axis=1)
        keep = (back_status.reshape(-1) == 1) & (errors <= self.fb_max_error)
        return old_points[keep], new_points[keep]

    def _valid_bbox_update(self, old_bbox: BBox, new_bbox: BBox) -> bool:
        old_area = bbox_area(old_bbox)
        new_area = bbox_area(new_bbox)
        if old_area <= 0 or new_area <= 0:
            return False
        ratio = new_area / old_area
        if ratio < self.min_area_ratio or ratio > self.max_area_ratio:
            return False
        old_aspect = old_bbox[2] / max(old_bbox[3], 1)
        new_aspect = new_bbox[2] / max(new_bbox[3], 1)
        aspect_ratio = new_aspect / max(old_aspect, 1e-6)
        return 0.5 <= aspect_ratio <= 2.0

    def _estimate_transform(
        self,
        old_points: np.ndarray,
        new_points: np.ndarray,
    ) -> np.ndarray | None:
        if not self.use_affine or len(new_points) < 3:
            return None
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                old_points,
                new_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
                maxIters=100,
                confidence=0.98,
                refineIters=5,
            )
        except cv2.error:
            return None
        if matrix is None or inliers is None:
            return None
        if int(inliers.sum()) < max(3, self.min_points // 2):
            return None
        return matrix.astype(np.float32)

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        moved = pts @ matrix[:, :2].T + matrix[:, 2]
        return moved.astype(np.float32)

    def refresh_points(self, gray: np.ndarray, track: Track) -> Track:
        track.points = self.seed_points(gray, track.bbox)
        return track

    @staticmethod
    def _miss(track: Track) -> Track:
        vx, vy = track.velocity
        x, y, w, h = track.bbox
        track.bbox = (x + int(round(vx)), y + int(round(vy)), w, h)
        if track.corners is not None:
            track.corners = track.corners + np.asarray([vx, vy], dtype=np.float32)
        track.missed += 1
        track.age += 1
        track.source = "predicted"
        track.history.append(bbox_center(track.bbox))
        track.history = track.history[-64:]
        return track
