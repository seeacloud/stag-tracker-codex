from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, StagCandidate, StagObservation, Track


STAG_TRACK_COLOR = (40, 220, 120)
HOLD_TRACK_COLOR = (255, 120, 40)
FLOW_TRACK_COLOR = (220, 80, 220)
TRACK_FALLBACK_COLOR = (80, 180, 220)
YOLO_COLOR = (255, 170, 40)
STAG_COLOR = STAG_TRACK_COLOR
CANDIDATE_COLOR = (255, 90, 30)  # BGR — bright blue for unrecognized quad candidates
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


def draw_candidates(
    frame: np.ndarray,
    candidates: list[StagCandidate],
    label: bool = True,
) -> np.ndarray:
    """Draw rejected STag candidate quads as thin blue boxes.

    These are quads the detector localized but could not decode into a known id —
    typically due to occlusion, glare, blur, or contrast. Showing them helps the
    user understand 'something is there but we couldn't read it'.
    """
    for cand in candidates:
        corners = cand.corners.astype(int).reshape(-1, 2)
        cv2.polylines(
            frame,
            [corners],
            isClosed=True,
            color=CANDIDATE_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        if label:
            x, y, _, _ = cand.bbox
            _label(frame, x, y, "stag?", CANDIDATE_COLOR)
    return frame


def draw_tracks(frame: np.ndarray, tracks: list[Track], visual_hold: int = 0) -> np.ndarray:
    for track in tracks:
        x, y, w, h = _display_bbox(track)
        visually_seen = track.detection_missed <= visual_hold
        state = track_display_state(track, visual_hold)
        color = track_color(track, visual_hold)

        if track.display_corners is not None and visually_seen:
            corners = track.display_corners.astype(int).reshape(-1, 2)
            cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=2)

        label = (
            f"#{track.track_id} {track.label} "
            f"{track.confidence:.2f} {state} "
            f"miss:{track.missed} seenmiss:{track.detection_missed}"
        )
        _label(frame, x, y, label, color)

        draw_track_history(frame, track, visual_hold)
    return frame


def track_display_state(track: Track, visual_hold: int = 0) -> str:
    if track.source == "stag":
        return "stag"
    if track.marker_id is not None and track.detection_missed <= visual_hold:
        return "hold"
    if track.source in {"flow", "predicted"}:
        return "flow"
    return track.source


def track_color(track: Track, visual_hold: int = 0) -> tuple[int, int, int]:
    state = track_display_state(track, visual_hold)
    return track_state_color(state)


def track_state_color(state: str) -> tuple[int, int, int]:
    if state == "stag":
        return STAG_TRACK_COLOR
    if state == "hold":
        return HOLD_TRACK_COLOR
    if state == "flow":
        return FLOW_TRACK_COLOR
    return TRACK_FALLBACK_COLOR


def track_history_state(track: Track, point_index: int, visual_hold: int = 0) -> str:
    source = _history_value(track.history_sources, point_index, track.source)
    detection_missed = _history_value(
        track.history_detection_missed,
        point_index,
        track.detection_missed,
    )
    marker_id = _history_value(track.history_marker_ids, point_index, track.marker_id)

    if source == "stag":
        return "stag"
    if marker_id is not None and detection_missed <= visual_hold:
        return "hold"
    if source in {"flow", "predicted"}:
        return "flow"
    return source


def track_history_segment_color(
    track: Track,
    point_index: int,
    visual_hold: int = 0,
) -> tuple[int, int, int]:
    return track_state_color(track_history_state(track, point_index, visual_hold))


def draw_track_history(
    frame: np.ndarray,
    track: Track,
    visual_hold: int = 0,
) -> None:
    if len(track.history) < 2:
        return
    pts = np.asarray(track.history, dtype=np.int32).reshape(-1, 2)
    for index in range(1, len(pts)):
        color = track_history_segment_color(track, index, visual_hold)
        cv2.line(
            frame,
            tuple(int(value) for value in pts[index - 1]),
            tuple(int(value) for value in pts[index]),
            color,
            1,
            cv2.LINE_AA,
        )


def _history_value(values: list, index: int, fallback):
    if 0 <= index < len(values):
        return values[index]
    return fallback


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
