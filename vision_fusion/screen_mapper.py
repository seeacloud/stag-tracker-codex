from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import BBox, StagObservation, Track, bbox_from_points, clip_bbox


@dataclass(slots=True)
class ScreenMapper:
    source_points: np.ndarray
    output_size: tuple[int, int]
    camera_to_screen: np.ndarray
    screen_to_camera: np.ndarray

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        output_size: tuple[int, int] | None = None,
    ) -> "ScreenMapper":
        src = np.asarray(points, dtype=np.float32).reshape(4, 2)
        if output_size is None:
            output_size = estimate_output_size(src)
        width, height = output_size
        dst = np.asarray(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        camera_to_screen = cv2.getPerspectiveTransform(src, dst)
        screen_to_camera = cv2.getPerspectiveTransform(dst, src)
        return cls(
            source_points=src,
            output_size=(int(width), int(height)),
            camera_to_screen=camera_to_screen,
            screen_to_camera=screen_to_camera,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScreenMapper":
        data = np.load(path)
        return cls.from_points(
            np.asarray(data["source_points"], dtype=np.float32),
            output_size=(
                int(np.asarray(data["output_size"])[0]),
                int(np.asarray(data["output_size"])[1]),
            ),
        )

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            source_points=self.source_points,
            output_size=np.asarray(self.output_size, dtype=np.int32),
            camera_to_screen=self.camera_to_screen,
            screen_to_camera=self.screen_to_camera,
        )

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.camera_to_screen, self.output_size)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.camera_to_screen)
        return mapped.reshape(-1, 2)

    def source_bbox(self, frame_shape: tuple[int, ...], padding: int = 0) -> BBox:
        height, width = frame_shape[:2]
        bbox = bbox_from_points(self.source_points)
        return clip_bbox(bbox, width, height, padding=padding)

    def draw_source_outline(self, frame: np.ndarray) -> None:
        points = self.source_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        for index, point in enumerate(self.source_points.astype(int)):
            cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
            cv2.putText(
                frame,
                str(index + 1),
                tuple(point + np.asarray([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )


def estimate_output_size(points: np.ndarray) -> tuple[int, int]:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    right = np.linalg.norm(pts[2] - pts[1])
    left = np.linalg.norm(pts[3] - pts[0])
    width = max(16, int(round(max(top, bottom))))
    height = max(16, int(round(max(left, right))))
    return width, height


def draw_screen_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    mapper: ScreenMapper,
    visual_hold: int = 0,
) -> None:
    for track in tracks:
        visually_seen = track.detection_missed <= visual_hold
        if track.display_corners is not None and visually_seen:
            mapped = mapper.transform_points(track.display_corners).astype(np.int32)
            cv2.polylines(frame, [mapped.reshape(-1, 1, 2)], True, (80, 160, 255), 2)
            label_point = mapped[0]
        else:
            x, y, w, h = track.display_bbox if track.display_bbox is not None else track.bbox
            corners = np.asarray(
                [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h],
                ],
                dtype=np.float32,
            )
            mapped = mapper.transform_points(corners).astype(np.int32)
            cv2.polylines(frame, [mapped.reshape(-1, 1, 2)], True, (80, 180, 220), 2)
            label_point = mapped[0]

        state = track.source if track.source == "stag" else ("hold" if visually_seen else track.source)
        label = f"#{track.track_id} {track.label} {state} seenmiss:{track.detection_missed}"
        draw_label(frame, int(label_point[0]), int(label_point[1]), label)


def draw_screen_observations(
    frame: np.ndarray,
    observations: list[StagObservation],
    mapper: ScreenMapper,
) -> None:
    for observation in observations:
        mapped = mapper.transform_points(observation.corners).astype(np.int32)
        cv2.polylines(frame, [mapped.reshape(-1, 1, 2)], True, (80, 160, 255), 2)
        label = f"stag:{observation.marker_id}"
        draw_label(frame, int(mapped[0][0]), int(mapped[0][1]), label)


def draw_label(frame: np.ndarray, x: int, y: int, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(frame.shape[1] - 1, x))
    y = max(th + 5, min(frame.shape[0] - 1, y))
    cv2.rectangle(
        frame,
        (x, y - th - baseline - 4),
        (min(frame.shape[1] - 1, x + tw + 6), y + baseline),
        (40, 40, 40),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, y - 3),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
