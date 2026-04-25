from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np

from .models import Detection


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        classes: Optional[Iterable[int]] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        half: bool = False,
        fuse: bool = False,
        max_det: int = 20,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_path)
        if fuse and hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception as exc:
                print(f"YOLO layer fuse skipped: {exc}")
        self.confidence = confidence
        self.classes = list(classes) if classes is not None else None
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.max_det = max_det

    def detect(self, frame: np.ndarray) -> list[Detection]:
        predict_kwargs = {
            "source": frame,
            "conf": self.confidence,
            "classes": self.classes,
            "device": self.device,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        if self.half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        names = getattr(results[0], "names", {}) or {}
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = xyxy
            w = max(0, int(round(x2 - x1)))
            h = max(0, int(round(y2 - y1)))
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))
            detections.append(
                Detection(
                    bbox=(int(round(x1)), int(round(y1)), w, h),
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
                )
            )
        return detections
