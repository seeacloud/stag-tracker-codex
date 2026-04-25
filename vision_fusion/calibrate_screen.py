from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .screen_mapper import ScreenMapper


POINT_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click four screen corners and save a perspective map.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--output", default="screen_map.npz", help="Output NPZ calibration path.")
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip frames horizontally before clicking points.",
    )
    parser.add_argument("--camera-width", type=int, default=None, help="Requested camera capture width.")
    parser.add_argument("--camera-height", type=int, default=None, help="Requested camera capture height.")
    parser.add_argument("--camera-fps", type=float, default=None, help="Requested camera capture FPS.")
    parser.add_argument("--camera-fourcc", default=None, help="Requested camera FOURCC, for example MJPG.")
    parser.add_argument(
        "--camera-backend",
        choices=("any", "dshow", "msmf"),
        default="any",
        help="OpenCV camera backend on Windows.",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional warped output width.")
    parser.add_argument("--height", type=int, default=None, help="Optional warped output height.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = open_capture(args)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Video source opened but did not return a frame.")
    frame = maybe_mirror(frame, args.mirror)

    state = {"points": [], "frame": frame.copy()}
    window = "Click screen corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse, state)

    print("Click corners in order: top-left, top-right, bottom-right, bottom-left.")
    print("Keys: r reset, s save after 4 points, q/Esc quit.")

    try:
        while True:
            ok, frame = capture.read()
            if ok:
                frame = maybe_mirror(frame, args.mirror)
                state["frame"] = frame.copy()
            preview = draw_preview(state["frame"], state["points"])
            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(20)
            if key in (27, ord("q"), ord("Q")):
                return 1
            if key in (ord("r"), ord("R")):
                state["points"].clear()
            if key in (ord("s"), ord("S")) and len(state["points"]) == 4:
                mapper = make_mapper(state["points"], args)
                mapper.save(args.output)
                print(f"Saved screen map: {Path(args.output).resolve()}")
                print(f"Output size: {mapper.output_size[0]}x{mapper.output_size[1]}")
                return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


def on_mouse(event: int, x: int, y: int, flags: int, state: dict) -> None:
    del flags
    points = state["points"]
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((float(x), float(y)))
        if len(points) < 4:
            print(f"Point {len(points)} set. Next: {POINT_NAMES[len(points)]}")
        else:
            print("Four points set. Press s to save, or r to reset.")


def draw_preview(frame: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    preview = frame.copy()
    for index, point in enumerate(points):
        p = tuple(int(v) for v in point)
        cv2.circle(preview, p, 6, (0, 255, 255), -1)
        cv2.putText(
            preview,
            f"{index + 1}:{POINT_NAMES[index]}",
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) >= 2:
        cv2.polylines(
            preview,
            [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)],
            isClosed=len(points) == 4,
            color=(0, 255, 255),
            thickness=2,
        )

    next_text = "Press s to save" if len(points) == 4 else f"Click {POINT_NAMES[len(points)]}"
    cv2.rectangle(preview, (8, 8), (520, 36), (40, 40, 40), -1)
    cv2.putText(
        preview,
        next_text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return preview


def make_mapper(points: list[tuple[float, float]], args: argparse.Namespace) -> ScreenMapper:
    output_size = None
    if args.width or args.height:
        if not args.width or not args.height:
            raise SystemExit("--width and --height must be provided together.")
        output_size = (args.width, args.height)
    return ScreenMapper.from_points(np.asarray(points, dtype=np.float32), output_size)


def maybe_mirror(frame, enabled: bool):
    if not enabled:
        return frame
    return cv2.flip(frame, 1)


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source.isdigit():
        capture = cv2.VideoCapture(int(args.source), camera_backend(args.camera_backend))
        configure_camera(capture, args)
        return capture
    return cv2.VideoCapture(args.source)


def camera_backend(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def configure_camera(capture: cv2.VideoCapture, args: argparse.Namespace) -> None:
    if args.camera_fourcc:
        if len(args.camera_fourcc) < 4:
            raise SystemExit("--camera-fourcc must have at least 4 characters, for example MJPG.")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc[:4]))
    if args.camera_width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


if __name__ == "__main__":
    raise SystemExit(main())
