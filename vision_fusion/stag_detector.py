from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
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


def _detect_scale_worker(
    shm_name: str,
    frame_shape: tuple[int, int, int],
    frame_dtype: str,
    roi: BBox,
    roi_padding: int,
    enhance_dict: dict,
    scale: float,
    library_hd: int,
) -> tuple[list[dict], list[dict]]:
    """One detectMarkers call: shared-memory frame → crop → enhance → scale → detect."""
    import stag as _stag

    shm = SharedMemory(name=shm_name, create=False)
    try:
        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
        height, width = frame.shape[:2]
        x, y, w, h = clip_bbox(roi, width, height, roi_padding)
        if w <= 4 or h <= 4:
            return [], []
        crop = frame[y : y + h, x : x + w].copy()
    finally:
        shm.close()

    enhance_cfg = EnhanceConfig(**enhance_dict)
    prepared = enhance_for_detection(crop, enhance_cfg)

    if abs(scale - 1.0) < 1e-3:
        scaled = prepared
    else:
        sh, sw = prepared.shape[:2]
        new_w = int(round(sw * scale))
        new_h = int(round(sh * scale))
        if new_w < 8 or new_h < 8:
            return [], []
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(prepared, (new_w, new_h), interpolation=interp)

    try:
        corners, ids, rejected = _stag.detectMarkers(scaled, libraryHD=library_hd)
    except TypeError:
        corners, ids, rejected = _stag.detectMarkers(scaled, library_hd)

    observations: list[dict] = []
    candidates: list[dict] = []
    inv_scale = 1.0 / scale if scale != 0 else 1.0

    if corners is not None and ids is not None:
        flat_ids = np.asarray(ids).reshape(-1)
        for raw_corners, marker_id in zip(corners, flat_ids):
            mc = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 2)
            if mc.shape[0] != 4:
                continue
            if scale != 1.0:
                mc *= inv_scale
            mc[:, 0] += x
            mc[:, 1] += y
            observations.append({
                "marker_id": int(marker_id),
                "corners": mc,
                "bbox": bbox_from_points(mc),
            })
    if rejected is not None:
        for raw_corners in rejected:
            quad = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 2)
            if quad.shape[0] != 4:
                continue
            if scale != 1.0:
                quad = quad * inv_scale
            quad[:, 0] += x
            quad[:, 1] += y
            candidates.append({
                "corners": quad,
                "bbox": bbox_from_points(quad),
            })
    return observations, candidates


def _effective_scales_static(
    short_side: int,
    scales: tuple[float, ...],
    roi_min_short_side: int,
) -> tuple[float, ...]:
    if roi_min_short_side <= 0:
        return scales
    if short_side <= 0 or short_side >= roi_min_short_side:
        return scales
    boost = roi_min_short_side / float(short_side)
    if boost <= 1.0:
        return scales
    boosted = tuple(s * boost for s in scales)
    seen: set[float] = set()
    merged: list[float] = []
    for s in (*scales, *boosted):
        key = round(s, 4)
        if key in seen:
            continue
        seen.add(key)
        merged.append(s)
    return tuple(merged)


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
        expected_ids: Optional[set[int]] = None,
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
        self.expected_ids: Optional[frozenset[int]] = (
            frozenset(int(i) for i in expected_ids) if expected_ids else None
        )
        self.last_skipped_passes: int = 0

        workers = max(1, int(pass_workers))
        self._pass_workers = workers
        self._executor: Optional[ProcessPoolExecutor] = (
            ProcessPoolExecutor(max_workers=workers)
            if workers > 1
            else None
        )
        self._shm: Optional[SharedMemory] = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_shm(self, frame: np.ndarray) -> SharedMemory:
        nbytes = frame.nbytes
        if self._shm is not None and self._shm.size >= nbytes:
            return self._shm
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
        self._shm = SharedMemory(create=True, size=nbytes)
        return self._shm

    def detect(
        self,
        frame: np.ndarray,
        rois: Optional[list[BBox]] = None,
    ) -> list[StagObservation]:
        height, width = frame.shape[:2]
        if rois is None or not rois:
            rois = [(0, 0, width, height)]

        self.last_skipped_passes = 0

        # Single-pass serial fast path
        if len(self.passes) == 1 and self._executor is None:
            crop_obs, crop_candidates = self._run_pass(frame, rois, self.passes[0])
            deduped = dedupe_observations(crop_obs)
            self.last_candidates = filter_candidates(crop_candidates, deduped)
            return deduped

        # Adaptive path: run baseline in-process first for fast skip check
        if self.expected_ids is not None:
            baseline_obs, baseline_cands = self._run_pass(frame, rois, self.passes[0])
            seen_ids = {obs.marker_id for obs in baseline_obs}
            if self.expected_ids.issubset(seen_ids):
                self.last_skipped_passes = len(self.passes) - 1
                deduped = dedupe_observations(baseline_obs)
                self.last_candidates = filter_candidates(baseline_cands, deduped)
                return deduped
            remaining_passes = self.passes[1:]
            baseline_results = [(baseline_obs, baseline_cands)]
        else:
            remaining_passes = self.passes
            baseline_results = []

        # Parallel path: shared memory + scale-level parallelism
        if self._executor is not None:
            shm = self._ensure_shm(frame)
            shm_arr = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
            np.copyto(shm_arr, frame)

            tasks = self._build_scale_tasks_for(frame, rois, remaining_passes)
            futures = [
                self._executor.submit(
                    _detect_scale_worker,
                    shm.name,
                    frame.shape,
                    str(frame.dtype),
                    roi,
                    self.roi_padding,
                    pass_cfg.enhance.to_dict(),
                    scale,
                    self.library_hd,
                )
                for roi, pass_cfg, scale in tasks
            ]
            raw_results = [f.result() for f in futures]
            pass_results = list(baseline_results)
            for obs_dicts, cand_dicts in raw_results:
                pass_results.append(self._deserialize_results(obs_dicts, cand_dicts))
        else:
            pass_results = list(baseline_results)
            for pass_cfg in remaining_passes:
                pass_results.append(self._run_pass(frame, rois, pass_cfg))

        return self._merge_pass_results(pass_results)

    def _build_scale_tasks_for(
        self,
        frame: np.ndarray,
        rois: list[BBox],
        passes: list[PassConfig],
    ) -> list[tuple[BBox, PassConfig, float]]:
        height, width = frame.shape[:2]
        tasks: list[tuple[BBox, PassConfig, float]] = []
        for roi in rois:
            x, y, w, h = clip_bbox(roi, width, height, self.roi_padding)
            if w <= 4 or h <= 4:
                continue
            short = min(w, h)
            for pass_cfg in passes:
                scales = _effective_scales_static(
                    short, pass_cfg.scales, pass_cfg.roi_min_short_side
                )
                for scale in scales:
                    tasks.append((roi, pass_cfg, scale))
        return tasks

    def _deserialize_results(
        self,
        obs_dicts: list[dict],
        cand_dicts: list[dict],
    ) -> tuple[list[StagObservation], list[StagCandidate]]:
        observations = [
            StagObservation(
                marker_id=d["marker_id"],
                corners=d["corners"],
                bbox=d["bbox"],
                pose=None,
            )
            for d in obs_dicts
        ]
        candidates = [
            StagCandidate(corners=d["corners"], bbox=d["bbox"])
            for d in cand_dicts
        ]
        return observations, candidates

    def _merge_pass_results(
        self,
        pass_results: list[tuple[list[StagObservation], list[StagCandidate]]],
    ) -> list[StagObservation]:
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

    def _run_pass(
        self,
        frame: np.ndarray,
        rois: list[BBox],
        pass_cfg: PassConfig,
    ) -> tuple[list[StagObservation], list[StagCandidate]]:
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
        short = min(prepared.shape[:2])
        scales = _effective_scales_static(
            short, pass_cfg.scales, pass_cfg.roi_min_short_side
        )
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
                image, libraryHD=self.library_hd,
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
            [[-half, half, 0.0], [half, half, 0.0],
             [half, -half, 0.0], [-half, -half, 0.0]],
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
        observations, key=lambda obs: bbox_area(obs.bbox), reverse=True,
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
