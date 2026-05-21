"""Label markers for YOLO training — manual 4-point marking with drag-to-adjust.

Usage:
    python -m vision_fusion.training.label_yolo --output yolo_data/ [--source 0]

Workflow per frame:
    1. Video pauses on a new frame
    2. Click 4 corners of each marker (can mark multiple markers per frame)
    3. Drag any point to adjust
    4. Press S to save frame + annotations, then auto-advances to next frame
    5. Press N to skip frame without saving

Outputs YOLO format:
    yolo_data/images/  — saved frames
    yolo_data/labels/  — "0 x_center y_center width height" (normalized)

Controls:
    Left click  - place corner point (every 4 points = one marker box)
    Drag        - adjust placed point
    S           - save current frame + all boxes, advance to next frame
    N           - skip frame, advance without saving
    Z           - undo last point or last completed box
    Q/ESC       - quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


class YoloLabeler:
    def __init__(self, output_dir: Path):
        self.img_dir = output_dir / "images"
        self.lbl_dir = output_dir / "labels"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        self.points: list[tuple[int, int]] = []
        self.boxes: list[tuple[int, int, int, int]] = []  # completed boxes (x1,y1,x2,y2)
        self.dragging_idx: int = -1
        self.frame = None
        self.display = None
        self.message = ""

        # Resume from existing count
        existing = list(self.img_dir.glob("*.jpg"))
        self.saved_count = len(existing)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit_test(x, y)
            if hit >= 0:
                self.dragging_idx = hit
            elif len(self.points) < 4:
                self.points.append((x, y))
                if len(self.points) == 4:
                    self.message = "4 points placed. Drag to adjust, or click more for next marker. S=save"
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx >= 0:
            self.points[self.dragging_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_idx >= 0:
                self.dragging_idx = -1
                # If we have 4 points and user finished dragging, finalize this box
                if len(self.points) == 4:
                    self._finalize_box()

    def _hit_test(self, x: int, y: int, radius: int = 14) -> int:
        for i, pt in enumerate(self.points):
            if abs(pt[0] - x) < radius and abs(pt[1] - y) < radius:
                return i
        return -1

    def _finalize_box(self):
        if len(self.points) != 4:
            return
        pts = np.array(self.points, dtype=np.int32)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            self.boxes.append((int(x1), int(y1), int(x2), int(y2)))
        self.points.clear()
        self.message = f"{len(self.boxes)} marker(s) marked. Click more or S=save"

    def save_frame(self) -> bool:
        if self.frame is None:
            return False
        # Finalize any pending 4-point set
        if len(self.points) == 4:
            self._finalize_box()
        if not self.boxes:
            self.message = "No boxes to save. Mark at least one marker."
            return False

        h, w = self.frame.shape[:2]
        self.saved_count += 1
        filename = f"frame_{self.saved_count:06d}"

        cv2.imwrite(str(self.img_dir / f"{filename}.jpg"), self.frame)
        with open(self.lbl_dir / f"{filename}.txt", "w") as f:
            for x1, y1, x2, y2 in self.boxes:
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        self.message = f"Saved #{self.saved_count} ({len(self.boxes)} boxes)"
        self.boxes.clear()
        self.points.clear()
        return True

    def undo(self):
        if self.points:
            self.points.pop()
            self.message = f"{len(self.points)}/4 points"
        elif self.boxes:
            self.boxes.pop()
            self.message = f"Undid box. {len(self.boxes)} remain"

    def draw_overlay(self):
        self.display = self.frame.copy()

        # Completed boxes (green)
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(self.display, f"#{i+1}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

        # In-progress points (colored circles, draggable)
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]
        for i, pt in enumerate(self.points):
            cv2.circle(self.display, pt, 7, colors[i % 4], -1)
            cv2.circle(self.display, pt, 9, (255, 255, 255), 1)
            if i > 0:
                cv2.line(self.display, self.points[i - 1], pt, (200, 200, 200), 1)
        if len(self.points) == 4:
            cv2.line(self.display, self.points[3], self.points[0], (200, 200, 200), 1)

        status = f"total_saved={self.saved_count} | boxes={len(self.boxes)} pts={len(self.points)}/4 | S=save+next, N=skip, Z=undo, Q=quit"
        cv2.putText(self.display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if self.message:
            cv2.putText(self.display, self.message, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label markers for YOLO training.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=60)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-backend", choices=("any", "dshow", "msmf"), default="msmf")
    args = parser.parse_args()

    from ..camera_picker import camera_backend

    output_dir = Path(args.output)
    source = int(args.source) if args.source.isdigit() else args.source
    backend = camera_backend(args.camera_backend)
    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if args.camera_fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))

    if not cap.isOpened():
        print(f"ERROR: Cannot open source {args.source}", file=sys.stderr)
        return 1

    tool = YoloLabeler(output_dir)
    if tool.saved_count > 0:
        print(f"Resuming: {tool.saved_count} frames already saved")

    cv2.namedWindow("YOLO Labeler")
    cv2.setMouseCallback("YOLO Labeler", tool.mouse_callback)

    print("Mark all markers in frame (4 clicks each), then S to save + next frame")
    print("N=skip frame, Z=undo, Q=quit")

    # Grab first frame
    ok, frame = cap.read()
    if not ok:
        print("ERROR: No frame from camera", file=sys.stderr)
        return 1
    tool.frame = frame
    tool.message = "Mark markers, then press S to save"

    try:
        while True:
            tool.draw_overlay()
            cv2.imshow("YOLO Labeler", tool.display)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                if tool.save_frame():
                    # Advance to next frame
                    ok, frame = cap.read()
                    if not ok:
                        print("End of video source")
                        break
                    tool.frame = frame
                    tool.message = "Saved! Mark next frame, then S"
            elif key == ord('n'):
                # Skip without saving
                ok, frame = cap.read()
                if not ok:
                    print("End of video source")
                    break
                tool.frame = frame
                tool.boxes.clear()
                tool.points.clear()
                tool.message = "Skipped. Mark this frame or N again"
            elif key == ord('z'):
                tool.undo()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Total saved: {tool.saved_count} annotated frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
