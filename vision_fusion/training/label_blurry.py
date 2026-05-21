"""Interactive 4-point labeling tool for blurry markers that STag can't locate.

Usage:
    python -m vision_fusion.training.label_blurry --output training_data/ [--source 0]

Controls:
    SPACE       - pause/resume video
    Left click  - place corner point (4 points = one quad)
    0-9         - type marker ID after placing 4 points
    Enter       - confirm ID and save patch
    Z           - undo last point (or clear all if quad complete)
    Q/ESC       - quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PATCH_SIZE = 128
WARP_PADDING = 24
TOTAL_SIZE = PATCH_SIZE + 2 * WARP_PADDING

DST_PTS = np.array([
    [WARP_PADDING, WARP_PADDING],
    [WARP_PADDING + PATCH_SIZE, WARP_PADDING],
    [WARP_PADDING + PATCH_SIZE, WARP_PADDING + PATCH_SIZE],
    [WARP_PADDING, WARP_PADDING + PATCH_SIZE],
], dtype=np.float32)

COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


class LabelingTool:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.points: list[tuple[int, int]] = []
        self.dragging_idx: int = -1  # which point is being dragged
        self.id_buffer = ""
        self.paused = False
        self.saved_count = 0
        self.frame = None
        self.display = None
        self.message = ""

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near an existing point (drag to adjust)
            hit = self._hit_test(x, y)
            if hit >= 0:
                self.dragging_idx = hit
            elif len(self.points) < 4:
                self.points.append((x, y))
                if len(self.points) == 4:
                    self.message = "4 points set. Type ID + Enter to save. Drag points to adjust."
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx >= 0:
            self.points[self.dragging_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = -1

    def _hit_test(self, x: int, y: int, radius: int = 12) -> int:
        """Return index of point near (x,y), or -1."""
        for i, pt in enumerate(self.points):
            if abs(pt[0] - x) < radius and abs(pt[1] - y) < radius:
                return i
        return -1

    def save_patch(self, frame_idx: int) -> bool:
        if len(self.points) != 4 or not self.id_buffer:
            return False
        try:
            marker_id = int(self.id_buffer)
        except ValueError:
            return False

        src_pts = np.array(self.points, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
        warped = cv2.warpPerspective(self.frame, M, (TOTAL_SIZE, TOTAL_SIZE))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped

        subdir = self.output_dir / f"{marker_id}_blurry"
        subdir.mkdir(parents=True, exist_ok=True)
        filename = f"manual_{frame_idx:06d}_{self.saved_count:04d}.png"
        cv2.imwrite(str(subdir / filename), gray)
        self.saved_count += 1
        self.message = f"Saved ID={marker_id} (#{self.saved_count})"
        self.points.clear()
        self.id_buffer = ""
        return True

    def undo(self):
        if self.id_buffer:
            self.id_buffer = ""
        elif self.points:
            self.points.pop()
            self.message = f"{len(self.points)}/4 points"

    def draw_overlay(self):
        self.display = self.frame.copy()

        # Draw placed points and connecting lines
        for i, pt in enumerate(self.points):
            color = COLORS[i]
            cv2.circle(self.display, pt, 6, color, -1)
            cv2.putText(self.display, CORNER_NAMES[i], (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if i > 0:
                cv2.line(self.display, self.points[i - 1], pt, (255, 255, 255), 1)
        if len(self.points) == 4:
            cv2.line(self.display, self.points[3], self.points[0], (255, 255, 255), 1)
            # Draw filled quad overlay
            pts_arr = np.array(self.points, dtype=np.int32)
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0, 50))
            cv2.addWeighted(overlay, 0.2, self.display, 0.8, 0, self.display)
            # Show ID input
            label = f"ID: {self.id_buffer}_" if self.id_buffer else "ID: _"
            cx = int(np.mean([p[0] for p in self.points]))
            cy = int(np.mean([p[1] for p in self.points]))
            cv2.putText(self.display, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status bar
        state = "PAUSED" if self.paused else "PLAYING"
        pts_status = f"points={len(self.points)}/4"
        status = f"saved={self.saved_count} | {state} | {pts_status} | SPACE=pause, click=corner, Z=undo, Enter=save"
        cv2.putText(self.display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if self.message:
            cv2.putText(self.display, self.message, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Label blurry markers with 4-point quad selection.")
    parser.add_argument("--output", required=True, help="Output directory for patches.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=60)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-backend", choices=("any", "dshow", "msmf"), default="msmf")
    args = parser.parse_args()

    from ..camera_picker import camera_backend

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    tool = LabelingTool(output_dir)
    cv2.namedWindow("Label Blurry Markers")
    cv2.setMouseCallback("Label Blurry Markers", tool.mouse_callback)

    frame_idx = 0
    print("4-point labeling: click TL, TR, BR, BL corners of blurry marker")
    print("Then type marker ID + Enter to save. SPACE=pause, Z=undo, Q=quit")

    try:
        while True:
            if not tool.paused:
                ok, frame = cap.read()
                if not ok:
                    break
                tool.frame = frame
                frame_idx += 1
            elif tool.frame is None:
                ok, frame = cap.read()
                if not ok:
                    break
                tool.frame = frame
                frame_idx += 1

            tool.draw_overlay()
            cv2.imshow("Label Blurry Markers", tool.display)

            key = cv2.waitKey(30 if not tool.paused else 50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                tool.paused = not tool.paused
            elif key == ord('z'):
                tool.undo()
            elif key == 13:  # Enter
                if tool.save_patch(frame_idx):
                    print(f"  Saved patch #{tool.saved_count}")
            elif ord('0') <= key <= ord('9'):
                if len(tool.points) == 4:
                    tool.id_buffer += chr(key)
            elif key == 8:  # Backspace
                tool.id_buffer = tool.id_buffer[:-1]
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Saved {tool.saved_count} manually labeled patches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
