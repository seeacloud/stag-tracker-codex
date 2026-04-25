from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, StagObservation, Track


TRACK_COLOR = (40, 220, 120)
YOLO_COLOR = (255, 170, 40)
STAG_COLOR = (80, 160, 255)
TEXT_COLOR = (245, 245, 245)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), YOLO_COLOR, 1)
        _label(frame, x, y, f"{det.label} {det.confidence:.2f}", YOLO_COLOR)
    return frame


def draw_observations(frame: np.ndarray, observations: list[StagObservation]) -> np.ndarray:
    for obs in observations:
        corners = obs.corners.astype(int).reshape(-1, 2)
        cv2.polylines(frame, [corners], isClosed=True, color=STAG_COLOR, thickness=2)
        x, y, _, _ = obs.bbox
        _label(frame, x, y, f"stag:{obs.marker_id}", STAG_COLOR)
        if obs.pose is not None:
            t = obs.pose.tvec.reshape(-1)
            _label(frame, x, y + 18, f"t=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})", STAG_COLOR)
    return frame


def draw_tracks(frame: np.ndarray, tracks: list[Track], visual_hold: int = 0) -> np.ndarray:
    for track in tracks:
        x, y, w, h = _display_bbox(track)
        visually_seen = track.detection_missed <= visual_hold
        color = TRACK_COLOR if visually_seen else (80, 180, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        state = track.source if track.source == "stag" else ("hold" if visually_seen else track.source)
        label = (
            f"#{track.track_id} {track.label} "
            f"{track.confidence:.2f} {state} "
            f"miss:{track.missed} seenmiss:{track.detection_missed}"
        )
        _label(frame, x, y, label, color)

        if track.display_corners is not None and visually_seen:
            corners = track.display_corners.astype(int).reshape(-1, 2)
            cv2.polylines(frame, [corners], isClosed=True, color=STAG_COLOR, thickness=2)

        if len(track.history) >= 2:
            pts = np.asarray(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=1)
    return frame


def _display_bbox(track: Track) -> tuple[int, int, int, int]:
    bbox = track.display_bbox if track.display_bbox is not None else track.bbox
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def draw_status(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    track_count: int,
    yolo_enabled: bool,
) -> np.ndarray:
    mode = "YOLO+STag+flow" if yolo_enabled else "STag+flow"
    text = f"{mode} | frame {frame_index} | fps {fps:.1f} | tracks {track_count}"
    _label(frame, 8, 22, text, (40, 40, 40), above=False)
    return frame


def _label(
    frame: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    above: bool = True,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = y - 6 if above else y
    y_text = max(th + 4, y_text)
    x = max(0, x)
    cv2.rectangle(
        frame,
        (x, y_text - th - baseline - 4),
        (x + tw + 6, y_text + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, y_text - 3),
        font,
        scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )
