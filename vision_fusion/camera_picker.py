from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int = 0
    height: int = 0
    sample: Optional[np.ndarray] = None


def camera_backend(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def list_device_names() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def probe_cameras(max_index: int = 8, backend: int = cv2.CAP_DSHOW) -> list[CameraInfo]:
    names = list_device_names()
    found: list[CameraInfo] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not ok or frame is None:
            continue
        name = names[index] if index < len(names) else f"Camera {index}"
        found.append(CameraInfo(index=index, name=name, width=width, height=height, sample=frame))
    return _sort_external_first(found)


_BUILT_IN_HINTS = ("integrated", "built-in", "builtin", "internal", "笔记本")


def _is_builtin(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in _BUILT_IN_HINTS)


def _sort_external_first(cameras: list[CameraInfo]) -> list[CameraInfo]:
    """USB / external cameras come first; built-in webcams last."""
    return sorted(cameras, key=lambda c: (_is_builtin(c.name), c.index))


def default_camera_index(cameras: list[CameraInfo]) -> Optional[int]:
    """First non-builtin camera, falling back to the first available."""
    if not cameras:
        return None
    for cam in cameras:
        if not _is_builtin(cam.name):
            return cam.index
    return cameras[0].index


def print_cameras(cameras: list[CameraInfo]) -> None:
    if not cameras:
        print("No cameras detected.")
        return
    print(f"{'idx':<5}{'resolution':<14}name")
    for cam in cameras:
        print(f"{cam.index:<5}{cam.width}x{cam.height:<8}{cam.name}")


def pick_camera_gui(cameras: list[CameraInfo], window: str = "Select camera") -> Optional[int]:
    if not cameras:
        return None
    if len(cameras) == 1:
        return cameras[0].index

    default_index = default_camera_index(cameras)
    cell_w, cell_h = 320, 240
    cols = min(2, len(cameras))
    rows = (len(cameras) + cols - 1) // cols
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    rects: list[tuple[int, int, int, int, int]] = []

    for i, cam in enumerate(cameras):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        rects.append((x, y, cell_w, cell_h, cam.index))
        thumb = cv2.resize(cam.sample, (cell_w, cell_h)) if cam.sample is not None else canvas[y:y + cell_h, x:x + cell_w]
        canvas[y:y + cell_h, x:x + cell_w] = thumb
        is_default = cam.index == default_index
        border = (60, 220, 60) if is_default else (90, 90, 90)
        thickness = 4 if is_default else 2
        cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), border, thickness)
        suffix = "  [default, press Enter]" if is_default else ""
        label = f"[{i + 1}] idx {cam.index}  {cam.name}{suffix}"
        sub = f"{cam.width}x{cam.height}"
        cv2.rectangle(canvas, (x, y + cell_h - 50), (x + cell_w, y + cell_h), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x + 10, y + cell_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub, (x + 10, y + cell_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)

    hint = "Click / press 1-9 / Enter for default / Esc to cancel"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (10, 10, 10), -1)
    cv2.putText(canvas, hint, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    chosen: dict[str, Optional[int]] = {"idx": None}

    def on_mouse(event: int, mx: int, my: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x, y, w, h, cam_index in rects:
            if x <= mx < x + w and y <= my < y + h:
                chosen["idx"] = cam_index
                return

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, canvas)

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key in (13, 10):
                if default_index is not None:
                    chosen["idx"] = default_index
                    break
            if ord("1") <= key <= ord("9"):
                slot = key - ord("1")
                if slot < len(cameras):
                    chosen["idx"] = cameras[slot].index
                    break
            if chosen["idx"] is not None:
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyWindow(window)

    return chosen["idx"]
