"""YOLO-based marker locator — finds marker regions even when severely blurred.

Replaces STag's edge-based quad detection as the first localization step.
Detected regions are warped and sent to the CNN classifier for ID assignment.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import BBox


class YoloLocator:
    """Locate markers using YOLOv8-nano trained on 'stag_marker' class."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        device: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._confidence = confidence
        self._device = device or "0"
        self._imgsz = imgsz

    def locate(
        self,
        frame: np.ndarray,
        existing_bboxes: Optional[list[BBox]] = None,
    ) -> list[tuple[np.ndarray, BBox]]:
        """Find marker-like regions in the frame.

        Returns list of (corners_4x2, bbox) for each detection.
        Skips detections that overlap with existing_bboxes.
        """
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        detections = []
        boxes = results[0].boxes

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox: BBox = (x1, y1, x2 - x1, y2 - y1)

            if existing_bboxes and self._center_in_any(bbox, existing_bboxes):
                continue

            corners = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)

            detections.append((corners, bbox))

        return detections

    @staticmethod
    def _center_in_any(bbox: BBox, existing: list[BBox]) -> bool:
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        for ex, ey, ew, eh in existing:
            if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
                return True
        return False
