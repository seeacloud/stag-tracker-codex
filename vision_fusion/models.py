from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


BBox = tuple[int, int, int, int]
TRACK_HISTORY_LIMIT = 64


@dataclass(slots=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    label: str


@dataclass(slots=True)
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass(slots=True)
class StagObservation:
    marker_id: int
    corners: np.ndarray
    bbox: BBox
    pose: Optional[Pose] = None


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    points: np.ndarray
    label: str
    confidence: float
    marker_id: Optional[int] = None
    corners: Optional[np.ndarray] = None
    display_bbox: Optional[tuple[float, float, float, float]] = None
    display_corners: Optional[np.ndarray] = None
    pose: Optional[Pose] = None
    missed: int = 0
    detection_missed: int = 0
    age: int = 0
    source: str = "init"
    velocity: tuple[float, float] = (0.0, 0.0)
    history: list[tuple[int, int]] = field(default_factory=list)
    history_sources: list[str] = field(default_factory=list)
    history_detection_missed: list[int] = field(default_factory=list)
    history_marker_ids: list[Optional[int]] = field(default_factory=list)


def clip_bbox(bbox: BBox, width: int, height: int, padding: int = 0) -> BBox:
    x, y, w, h = bbox
    x1 = max(0, int(round(x - padding)))
    y1 = max(0, int(round(y - padding)))
    x2 = min(width, int(round(x + w + padding)))
    y2 = min(height, int(round(y + h + padding)))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[int, int]:
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def append_track_history(
    track: Track,
    source: Optional[str] = None,
    detection_missed: Optional[int] = None,
    marker_id: Optional[int] = None,
) -> None:
    history_source = track.source if source is None else source
    missed = track.detection_missed if detection_missed is None else detection_missed
    tracked_marker_id = track.marker_id if marker_id is None else marker_id

    _pad_track_history_metadata(track, history_source, missed, tracked_marker_id)
    track.history.append(bbox_center(track.bbox))
    track.history_sources.append(history_source)
    track.history_detection_missed.append(missed)
    track.history_marker_ids.append(tracked_marker_id)
    trim_track_history(track)


def trim_track_history(track: Track, limit: int = TRACK_HISTORY_LIMIT) -> None:
    for values in (
        track.history,
        track.history_sources,
        track.history_detection_missed,
        track.history_marker_ids,
    ):
        overflow = len(values) - limit
        if overflow > 0:
            del values[:overflow]


def _pad_track_history_metadata(
    track: Track,
    source: str,
    detection_missed: int,
    marker_id: Optional[int],
) -> None:
    missing = len(track.history) - len(track.history_sources)
    if missing > 0:
        track.history_sources.extend([source] * missing)

    missing = len(track.history) - len(track.history_detection_missed)
    if missing > 0:
        track.history_detection_missed.extend([detection_missed] * missing)

    missing = len(track.history) - len(track.history_marker_ids)
    if missing > 0:
        track.history_marker_ids.extend([marker_id] * missing)


def bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2]) * max(0, bbox[3])


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_from_points(points: np.ndarray) -> BBox:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x, y, w, h = cv_bounding_rect(pts)
    return int(x), int(y), int(w), int(h)


def cv_bounding_rect(points: np.ndarray) -> BBox:
    import cv2

    x, y, w, h = cv2.boundingRect(np.asarray(points, dtype=np.float32))
    return int(x), int(y), int(w), int(h)
