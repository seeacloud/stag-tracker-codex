from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .models import (
    BBox,
    Pose,
    StagCandidate,
    StagObservation,
    bbox_area,
    bbox_from_points,
    bbox_iou,
    clip_bbox,
)
from .preprocess import EnhanceConfig, enhance_for_detection


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


@dataclass(slots=True)
class PassConfig:
    """One detection pass: a full set of preprocessing + scaling parameters.

    Multi-pass detection re-runs STag on the same frame with each PassConfig
    in order and unions the results. Use it to combine a fast 'normal' pass
    with one or more aggressive passes that recover dim / small markers.
    """

    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    scales: tuple[float, ...] = (1.0,)
    roi_min_short_side: int = 0


class StagDetector:
    def __init__(
        self,
        library_hd: int = 17,
        marker_size: Optional[float] = None,
        calibration: Optional[CameraCalibration] = None,
        roi_padding: int = 12,
        enhance: Optional[EnhanceConfig] = None,
        scales: Optional[tuple[float, ...]] = None,
        roi_min_short_side: int = 0,
        passes: Optional[list[PassConfig]] = None,
        pass_workers: int = 1,
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
        if passes:
            self.passes = list(passes)
        else:
            self.passes = [
                PassConfig(
                    enhance=enhance or EnhanceConfig(),
                    scales=tuple(scales) if scales else (1.0,),
                    roi_min_short_side=max(0, int(roi_min_short_side)),
                )
            ]
        self.last_candidates: list[StagCandidate] = []

        workers = max(1, int(pass_workers))
        # Cap at len(passes) — extra workers can't help since each pass is one task.
        self._pass_workers = min(workers, len(self.passes))
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self._pass_workers, thread_name_prefix="stag-pass")
            if self._pass_workers > 1
            else None
        )

    def close(self) -> None:
        """Tear down the per-pass thread pool, if any. Idempotent."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _run_pass(
        self,
        frame: np.ndarray,
        rois: list[BBox],
        pass_cfg: PassConfig,
    ) -> tuple[list[StagObservation], list[StagCandidate]]:
        """Run a single PassConfig across all rois. Pure function — safe to parallelise."""
        height, width = frame.shape[:2]
        observations: list[StagObservation] = []
        candidates: list[StagCandidate] = []
        for roi in rois:
            x, y, w, h = clip_bbox(roi, width, height, self.roi_padding)
            if w <= 4 or h <= 4:
                continue
            crop = frame[y : y + h, x : x + w]
            crop_obs, crop_cand = self._detect_crop(crop, x, y, pass_cfg)
            observations.extend(crop_obs)
            candidates.extend(crop_cand)
        return observations, candidates

    def detect(
        self,
        frame: np.ndarray,
        rois: Optional[list[BBox]] = None,
    ) -> list[StagObservation]:
        height, width = frame.shape[:2]
        if rois is None or not rois:
            rois = [(0, 0, width, height)]

        if self._executor is not None and len(self.passes) > 1:
            futures = [
                self._executor.submit(self._run_pass, frame, rois, pass_cfg)
                for pass_cfg in self.passes
            ]
            pass_results = [f.result() for f in futures]
        else:
            pass_results = [self._run_pass(frame, rois, pass_cfg) for pass_cfg in self.passes]

        observations: list[StagObservation] = []
        candidates: list[StagCandidate] = []
        seen: set[tuple[int, int, int]] = set()
        for crop_obs, crop_candidates in pass_results:
            for obs in crop_obs:
                key = (obs.marker_id, int(obs.bbox[0]), int(obs.bbox[1]))
                if key in seen:
                    continue
                seen.add(key)
                observations.append(obs)
            candidates.extend(crop_candidates)

        deduped = dedupe_observations(observations)
        self.last_candidates = filter_candidates(candidates, deduped)
        return deduped

    def _detect_crop(
        self,
        crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        pass_cfg: PassConfig,
    ) -> tuple[list[StagObservation], list[StagCandidate]]:
        prepared = enhance_for_detection(crop, pass_cfg.enhance)
        observations: list[StagObservation] = []
        candidates: list[StagCandidate] = []
        scales = self._effective_scales(prepared, pass_cfg)
        for scale in scales:
            scaled = self._scale_image(prepared, scale)
            if scaled is None:
                continue
            corners, ids, rejected = self._call_stag(scaled)
            if corners is not None and ids is not None:
                observations.extend(
                    self._build_observations(corners, ids, offset_x, offset_y, scale)
                )
            if rejected is not None:
                candidates.extend(
                    self._build_candidates(rejected, offset_x, offset_y, scale)
                )
        return observations, candidates

    def _effective_scales(
        self,
        image: np.ndarray,
        pass_cfg: PassConfig,
    ) -> tuple[float, ...]:
        base = pass_cfg.scales
        if pass_cfg.roi_min_short_side <= 0:
            return base
        h, w = image.shape[:2]
        short = min(h, w)
        if short <= 0 or short >= pass_cfg.roi_min_short_side:
            return base
        boost = pass_cfg.roi_min_short_side / float(short)
        if boost <= 1.0:
            return base
        boosted = tuple(s * boost for s in base)
        # Keep duplicates rare while preserving ordering.
        seen: set[float] = set()
        merged: list[float] = []
        for s in (*base, *boosted):
            key = round(s, 4)
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
        return tuple(merged)

    @staticmethod
    def _scale_image(image: np.ndarray, scale: float) -> Optional[np.ndarray]:
        if abs(scale - 1.0) < 1e-3:
            return image
        h, w = image.shape[:2]
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w < 8 or new_h < 8:
            return None
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    def _build_observations(
        self,
        corners,
        ids,
        offset_x: int,
        offset_y: int,
        scale: float,
    ) -> list[StagObservation]:
        observations: list[StagObservation] = []
        flat_ids = np.asarray(ids).reshape(-1)
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        for raw_corners, marker_id in zip(corners, flat_ids):
            marker_corners = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 2)
            if marker_corners.shape[0] != 4:
                continue
            if scale != 1.0:
                marker_corners *= inv_scale
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

    def _call_stag(self, image: np.ndarray) -> tuple[object, object, object]:
        try:
            corners, ids, rejected = self._stag.detectMarkers(
                image,
                libraryHD=self.library_hd,
            )
            return corners, ids, rejected
        except TypeError:
            corners, ids, rejected = self._stag.detectMarkers(image, self.library_hd)
            return corners, ids, rejected

    def _build_candidates(
        self,
        rejected,
        offset_x: int,
        offset_y: int,
        scale: float,
    ) -> list[StagCandidate]:
        candidates: list[StagCandidate] = []
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        for raw_corners in rejected:
            quad = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 2)
            if quad.shape[0] != 4:
                continue
            if scale != 1.0:
                quad = quad * inv_scale
            quad[:, 0] += offset_x
            quad[:, 1] += offset_y
            candidates.append(
                StagCandidate(corners=quad, bbox=bbox_from_points(quad))
            )
        return candidates

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


def filter_candidates(
    candidates: list[StagCandidate],
    observations: list[StagObservation],
    obs_iou_threshold: float = 0.3,
    self_iou_threshold: float = 0.4,
    min_short_side: int = 12,
) -> list[StagCandidate]:
    """Drop rejected quads that overlap a successful detection or each other."""
    filtered: list[StagCandidate] = []
    for cand in sorted(candidates, key=lambda c: bbox_area(c.bbox), reverse=True):
        x, y, w, h = cand.bbox
        if min(w, h) < min_short_side:
            continue
        if any(bbox_iou(cand.bbox, obs.bbox) > obs_iou_threshold for obs in observations):
            continue
        if any(bbox_iou(cand.bbox, kept.bbox) > self_iou_threshold for kept in filtered):
            continue
        filtered.append(cand)
    return filtered
