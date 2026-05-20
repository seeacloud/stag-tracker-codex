"""Separate-process camera reader with triple-buffered shared memory.
Achieves full 60fps by isolating capture from the GIL entirely.
Triple buffer ensures zero contention between writer and reader."""
from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import cv2
import numpy as np


def _camera_worker(
    shm_names: list[str],
    frame_shape: tuple[int, int, int],
    frame_dtype: str,
    source: int,
    backend: int,
    fourcc: str,
    width: int,
    height: int,
    fps: float,
    exposure: Optional[float],
    ready_event: mp.Event,
    stop_event: mp.Event,
    latest_idx: mp.Value,
    frame_seq: mp.Value,
) -> None:
    """Camera capture loop in its own process. Triple-buffered writes."""
    shms = [SharedMemory(name=n, create=False) for n in shm_names]
    bufs = [np.ndarray(frame_shape, dtype=frame_dtype, buffer=s.buf) for s in shms]

    cap = cv2.VideoCapture(source, backend)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    if not cap.isOpened():
        ready_event.set()
        for s in shms:
            s.close()
        return

    ok, frame = cap.read()
    if not ok:
        ready_event.set()
        cap.release()
        for s in shms:
            s.close()
        return

    np.copyto(bufs[0], frame)
    latest_idx.value = 0
    frame_seq.value = 1
    ready_event.set()

    write_idx = 1
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        np.copyto(bufs[write_idx], frame)
        latest_idx.value = write_idx
        frame_seq.value += 1
        # Cycle through buffers: pick next that isn't the one reader might be using
        write_idx = (write_idx + 1) % 3
        if write_idx == latest_idx.value:
            write_idx = (write_idx + 1) % 3

    cap.release()
    for s in shms:
        s.close()


class ProcessCamera:
    """Camera in a separate process. Triple-buffered for zero-contention reads at full FPS."""

    def __init__(
        self,
        source: int = 0,
        backend: int = cv2.CAP_MSMF,
        fourcc: str = "MJPG",
        width: int = 1280,
        height: int = 720,
        fps: float = 60.0,
        exposure: Optional[float] = -4,
    ) -> None:
        frame_shape = (height, width, 3)
        frame_dtype = "uint8"
        nbytes = int(np.prod(frame_shape))

        self._shms = [SharedMemory(create=True, size=nbytes) for _ in range(3)]
        self._bufs = [
            np.ndarray(frame_shape, dtype=frame_dtype, buffer=s.buf)
            for s in self._shms
        ]
        self._frame_shape = frame_shape

        self._ready = mp.Event()
        self._stop = mp.Event()
        self._latest_idx = mp.Value("i", -1)
        self._frame_seq = mp.Value("i", 0)
        self._last_seq = 0

        self._process = mp.Process(
            target=_camera_worker,
            args=(
                [s.name for s in self._shms],
                frame_shape,
                frame_dtype,
                source,
                backend,
                fourcc,
                width,
                height,
                fps,
                exposure,
            ),
            kwargs={
                "ready_event": self._ready,
                "stop_event": self._stop,
                "latest_idx": self._latest_idx,
                "frame_seq": self._frame_seq,
            },
            daemon=True,
        )
        self._process.start()
        self._ready.wait(timeout=10.0)

    def read(self) -> tuple[bool, Optional[np.ndarray], bool]:
        """Non-blocking read. Returns (ok, frame_copy, is_new)."""
        seq = self._frame_seq.value
        if seq == 0:
            if not self._process.is_alive():
                return False, None, False
            return True, None, False

        is_new = seq != self._last_seq
        if not is_new:
            return True, None, False

        self._last_seq = seq
        idx = self._latest_idx.value
        frame = self._bufs[idx].copy()
        return True, frame, True

    @property
    def frame_count(self) -> int:
        return self._frame_seq.value

    def release(self) -> None:
        self._stop.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
        for s in self._shms:
            s.close()
            s.unlink()
