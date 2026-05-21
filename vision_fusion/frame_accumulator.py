"""Multi-frame temporal accumulation for noise reduction.

At 60fps, averaging N aligned frames reduces noise by sqrt(N).
This gives STag cleaner edges to detect markers in noisy IR camera feeds.
"""

from __future__ import annotations

import numpy as np


class FrameAccumulator:
    """Accumulate aligned frames for noise reduction in marker ROIs.

    Uses a simple ring buffer of grayscale frames. No optical flow alignment
    is needed because at 60fps with mostly-static markers, simple pixel-wise
    averaging works well. Motion causes slight blur but moving markers have
    sharper edges anyway.

    Parameters
    ----------
    n_frames : int
        Number of frames to accumulate. Default 4 gives 2x noise reduction
        with only 67ms latency at 60fps.
    """

    def __init__(self, n_frames: int = 4) -> None:
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self._n_frames = n_frames
        self._buffer: list[np.ndarray] = []  # ring buffer of float32 frames
        self._index = 0  # next write position (wraps around)
        self._full = False  # whether buffer has been filled at least once

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def count(self) -> int:
        """Number of frames currently in the buffer."""
        return self._n_frames if self._full else self._index

    def add_frame(self, gray: np.ndarray) -> None:
        """Add a new grayscale frame to the ring buffer.

        Parameters
        ----------
        gray : np.ndarray
            Single-channel uint8 grayscale frame.
        """
        frame_f32 = gray.astype(np.float32)

        if len(self._buffer) < self._n_frames:
            # Still filling the buffer
            self._buffer.append(frame_f32)
            self._index = len(self._buffer)
            if self._index == self._n_frames:
                self._full = True
                self._index = 0
        else:
            # Overwrite oldest frame
            self._buffer[self._index] = frame_f32
            self._index = (self._index + 1) % self._n_frames

    def get_accumulated(self) -> np.ndarray:
        """Return the averaged frame (full frame, simple average of buffer).

        Returns uint8 grayscale. If buffer is empty, returns a zero frame
        (should not happen in normal usage since add_frame is called first).
        """
        n = self.count
        if n == 0:
            raise RuntimeError("No frames in accumulator buffer")
        if n == 1:
            return self._buffer[0].astype(np.uint8)

        # Stack and average
        stacked = np.stack(self._buffer[:n], axis=0)
        averaged = stacked.mean(axis=0)
        return averaged.astype(np.uint8)

    def get_accumulated_roi(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Return accumulated ROI region only.

        Parameters
        ----------
        bbox : tuple[int, int, int, int]
            (x, y, w, h) region of interest.

        Returns
        -------
        np.ndarray
            Averaged ROI as uint8 grayscale.
        """
        x, y, w, h = bbox
        n = self.count
        if n == 0:
            raise RuntimeError("No frames in accumulator buffer")
        if n == 1:
            return self._buffer[0][y:y + h, x:x + w].astype(np.uint8)

        # Average only the ROI region for efficiency
        roi_sum = np.zeros((h, w), dtype=np.float32)
        for i in range(n):
            roi_sum += self._buffer[i][y:y + h, x:x + w]
        roi_sum /= n
        return roi_sum.astype(np.uint8)
