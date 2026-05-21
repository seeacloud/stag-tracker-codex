"""Temporal voting to suppress false-positive (ghost) marker detections.

Requires a marker to be detected at roughly the same location in N out of M
recent frames before confirming it as a real detection. This eliminates
single-frame CNN misclassifications without adding perceptible latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class _VoteRecord:
    """Sliding window of recent detection frames for one marker id."""
    frame_indices: deque = field(default_factory=lambda: deque())
    centers: deque = field(default_factory=lambda: deque())
    confirmed: bool = False


class TemporalVoter:
    """Require consistent detection across multiple frames before confirming.

    Parameters
    ----------
    window : int
        Number of recent frames to consider (sliding window size).
    min_votes : int
        Minimum detections within the window to confirm a marker.
    max_drift_px : float
        Maximum distance (pixels) between bbox centers for detections to
        count as "same location".
    """

    def __init__(
        self,
        window: int = 8,
        min_votes: int = 3,
        max_drift_px: float = 20.0,
    ) -> None:
        self._window = window
        self._min_votes = min_votes
        self._max_drift_px = max_drift_px
        self._records: dict[int, _VoteRecord] = {}

    def submit(self, marker_id: int, bbox: tuple, frame_idx: int) -> bool:
        """Submit a detection. Returns True if confirmed (enough votes).

        Parameters
        ----------
        marker_id : int
            The detected marker id.
        bbox : tuple
            (x, y, w, h) bounding box of the detection.
        frame_idx : int
            Current frame index (monotonically increasing).
        """
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        rec = self._records.get(marker_id)
        if rec is None:
            rec = _VoteRecord()
            self._records[marker_id] = rec

        # Expire old entries outside the window
        cutoff = frame_idx - self._window
        while rec.frame_indices and rec.frame_indices[0] < cutoff:
            rec.frame_indices.popleft()
            rec.centers.popleft()

        # If the marker was confirmed but disappeared for a full window, reset
        if rec.confirmed and not rec.frame_indices:
            rec.confirmed = False

        # Add current detection
        rec.frame_indices.append(frame_idx)
        rec.centers.append((cx, cy))

        # Count spatially consistent votes
        votes = self._count_consistent_votes(rec, cx, cy)

        if votes >= self._min_votes:
            rec.confirmed = True

        return rec.confirmed

    def is_confirmed(self, marker_id: int) -> bool:
        """Check if a marker has enough temporal votes."""
        rec = self._records.get(marker_id)
        return rec is not None and rec.confirmed

    def _count_consistent_votes(
        self, rec: _VoteRecord, cx: float, cy: float
    ) -> int:
        """Count detections within max_drift_px of the given center."""
        threshold_sq = self._max_drift_px * self._max_drift_px
        count = 0
        for (px, py) in rec.centers:
            dx = px - cx
            dy = py - cy
            if dx * dx + dy * dy <= threshold_sq:
                count += 1
        return count
