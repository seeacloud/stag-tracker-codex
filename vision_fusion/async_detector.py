"""Async detection pipeline: decouple detection from the display loop.

Main thread submits frames for detection without blocking. Results are
consumed when ready. This lets the display loop run at camera FPS while
detection runs continuously in background processes.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from .models import BBox, StagCandidate, StagObservation
from .stag_detector import (
    PassConfig,
    _detect_scale_worker,
    _effective_scales_static,
    clip_bbox,
    dedupe_observations,
    filter_candidates,
)


class AsyncDetector:
    """Non-blocking detection wrapper around StagDetector.

    Usage:
        async_det = AsyncDetector(passes, workers=6, ...)
        # In main loop:
        async_det.submit(frame, rois)          # non-blocking
        results = async_det.try_get_results()  # non-blocking, None or (obs, cands)
    """

    def __init__(
        self,
        library_hd: int = 17,
        roi_padding: int = 12,
        passes: Optional[list[PassConfig]] = None,
        workers: int = 6,
    ) -> None:
        self.library_hd = library_hd
        self.roi_padding = roi_padding
        self.passes = passes or [PassConfig()]
        self._workers = max(1, workers)
        self._pool = mp.Pool(processes=self._workers)
        self._shm: Optional[SharedMemory] = None
        self._pending: list = []
        self._frame_shape: Optional[tuple] = None
        self._frame_dtype: Optional[str] = None

    def submit(self, frame: np.ndarray, rois: list[BBox]) -> None:
        """Submit a frame for async detection. Non-blocking.

        If previous detection is still in-flight, drop this frame.
        """
        if self._pending:
            if not all(r.ready() for r in self._pending):
                return
            self._pending = []

        self._frame_shape = frame.shape
        self._frame_dtype = str(frame.dtype)

        shm = self._ensure_shm(frame)
        shm_arr = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np.copyto(shm_arr, frame)

        tasks = self._expand_tasks(frame, rois)
        if not tasks:
            return

        self._pending = [
            self._pool.apply_async(
                _detect_scale_worker,
                (
                    shm.name,
                    frame.shape,
                    self._frame_dtype,
                    roi,
                    self.roi_padding,
                    pass_cfg.enhance.to_dict(),
                    scale,
                    self.library_hd,
                ),
            )
            for roi, pass_cfg, scale in tasks
        ]

    def try_get_results(self) -> Optional[tuple[list[StagObservation], list[StagCandidate]]]:
        """Non-blocking poll. Returns (observations, candidates) or None."""
        if not self._pending:
            return None

        if not all(r.ready() for r in self._pending):
            return None

        raw_observations: list[StagObservation] = []
        raw_candidates: list[StagCandidate] = []

        for result in self._pending:
            obs_dicts, cand_dicts = result.get()
            for d in obs_dicts:
                raw_observations.append(StagObservation(
                    marker_id=d["marker_id"],
                    corners=d["corners"],
                    bbox=d["bbox"],
                    pose=None,
                ))
            for d in cand_dicts:
                raw_candidates.append(StagCandidate(
                    corners=d["corners"],
                    bbox=d["bbox"],
                ))

        self._pending = []

        observations = dedupe_observations(raw_observations)
        candidates = filter_candidates(raw_candidates, observations)
        return observations, candidates

    def _expand_tasks(
        self, frame: np.ndarray, rois: list[BBox]
    ) -> list[tuple[BBox, PassConfig, float]]:
        """Expand (rois × passes × scales) into individual worker tasks.

        ROIs are passed raw — the worker handles clip+pad so there's no
        double-padding.
        """
        height, width = frame.shape[:2]
        tasks: list[tuple[BBox, PassConfig, float]] = []
        for roi in rois:
            x, y, w, h = clip_bbox(roi, width, height, self.roi_padding)
            if w <= 4 or h <= 4:
                continue
            short = min(w, h)
            for pass_cfg in self.passes:
                scales = _effective_scales_static(
                    short, pass_cfg.scales, pass_cfg.roi_min_short_side
                )
                for scale in scales:
                    tasks.append((roi, pass_cfg, scale))
        return tasks

    def _ensure_shm(self, frame: np.ndarray) -> SharedMemory:
        nbytes = frame.nbytes
        if self._shm is not None and self._shm.size >= nbytes:
            return self._shm
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
        self._shm = SharedMemory(create=True, size=nbytes)
        return self._shm

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
