from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2

from .fusion import FusionTracker
from .models import BBox, clip_bbox
from .optical_flow import OpticalFlowTracker
from .stag_detector import CameraCalibration, StagDetector
from .visualization import draw_detections, draw_observations, draw_status, draw_tracks
from .yolo_detector import YoloDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse YOLO detection, optical-flow tracking, and STag recognition.",
    )
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name.")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO and run STag on full frame.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional YOLO class IDs.")
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu, 0, cuda:0.")
    parser.add_argument("--yolo-imgsz", type=int, default=None, help="YOLO inference image size.")
    parser.add_argument("--yolo-half", action="store_true", help="Use FP16 YOLO inference on compatible GPUs.")
    parser.add_argument("--yolo-fuse", action="store_true", help="Fuse YOLO model layers before inference.")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum YOLO detections per frame.")
    parser.add_argument("--detect-interval", type=int, default=5, help="Run YOLO/STag refresh every N frames.")
    parser.add_argument("--stag-library", type=int, default=17, help="STag HD library number.")
    parser.add_argument("--roi-padding", type=int, default=12, help="Pixels added around candidate ROIs.")
    parser.add_argument("--marker-size", type=float, default=None, help="Marker side length in meters.")
    parser.add_argument("--calibration", default=None, help="NPZ with camera_matrix and dist_coeffs.")
    parser.add_argument("--max-missed", type=int, default=20, help="Drop tracks after this many missed frames.")
    parser.add_argument("--show", action="store_true", help="Show annotated frames.")
    parser.add_argument("--output", default=None, help="Optional annotated output video path.")
    parser.add_argument("--draw-raw", action="store_true", help="Draw raw YOLO boxes and STag observations.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 runs until the source ends.")
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
    parser.add_argument("--opencv-threads", type=int, default=0, help="OpenCV worker threads. 0 keeps OpenCV default.")
    parser.add_argument("--log-every", type=int, default=120, help="Print average FPS every N frames. 0 disables logs.")
    parser.add_argument("--fast60", action="store_true", help="Apply an aggressive 60 FPS-oriented preset.")
    args = parser.parse_args()
    apply_fast60_preset(args)
    return args


def apply_fast60_preset(args: argparse.Namespace) -> None:
    if not args.fast60:
        return
    if args.camera_fps is None:
        args.camera_fps = 60
    if args.camera_width is None:
        args.camera_width = 1280
    if args.camera_height is None:
        args.camera_height = 720
    if args.camera_fourcc is None:
        args.camera_fourcc = "MJPG"
    if args.yolo_imgsz is None:
        args.yolo_imgsz = 416
    args.detect_interval = max(args.detect_interval, 10)
    args.max_det = min(args.max_det, 10)
    args.yolo_fuse = True


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source = args.source
    if source.isdigit():
        capture = cv2.VideoCapture(int(source), camera_backend(args.camera_backend))
        configure_camera(capture, args)
        return capture
    return cv2.VideoCapture(source)


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
        fourcc = cv2.VideoWriter_fourcc(*args.camera_fourcc[:4])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    if args.camera_width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def build_writer(path: str, capture: cv2.VideoCapture, width: int, height: int) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def main() -> int:
    args = parse_args()
    configure_opencv(args)

    capture = open_capture(args)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Video source opened but did not return a frame.")

    height, width = frame.shape[:2]
    writer = build_writer(args.output, capture, width, height) if args.output else None

    yolo = None
    if not args.no_yolo:
        yolo = YoloDetector(
            model_path=args.yolo_model,
            confidence=args.conf,
            classes=args.classes,
            device=args.device,
            imgsz=args.yolo_imgsz,
            half=args.yolo_half,
            fuse=args.yolo_fuse,
            max_det=args.max_det,
        )

    calibration = load_calibration(args.calibration)
    stag = StagDetector(
        library_hd=args.stag_library,
        marker_size=args.marker_size,
        calibration=calibration,
        roi_padding=args.roi_padding,
    )
    flow = OpticalFlowTracker()
    fusion = FusionTracker(flow=flow, max_missed=args.max_missed)

    previous_gray = None
    frame_index = 0
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    fps_window: deque[float] = deque(maxlen=120)

    try:
        while ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = []
            observations = []
            predicted = False
            updated = False

            if previous_gray is not None and fusion.tracks:
                fusion.predict(previous_gray, gray)
                predicted = True

            should_refresh = frame_index % max(1, args.detect_interval) == 0
            if should_refresh:
                if yolo is not None:
                    detections = yolo.detect(frame)
                    rois = [det.bbox for det in detections]
                    observations = stag.detect(frame, rois=rois) if rois else []
                else:
                    rois = [full_frame_roi(frame)]
                    observations = stag.detect(frame, rois=rois)
                fusion.update(gray, detections, observations)
                updated = True

            if predicted and not updated:
                fusion.record_predicted_history()

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_window.append(fps)

            annotated = frame.copy()
            if args.draw_raw:
                draw_detections(annotated, detections)
                draw_observations(annotated, observations)
            draw_tracks(annotated, fusion.tracks)
            draw_status(annotated, frame_index, fps, len(fusion.tracks), yolo is not None)

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("YOLO + optical flow + STag", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            previous_gray = gray
            ok, frame = capture.read()
            frame_index += 1
            if args.log_every and frame_index % args.log_every == 0 and fps_window:
                avg_fps = sum(fps_window) / len(fps_window)
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                wall_fps = frame_index / elapsed
                print(
                    f"frame={frame_index} fps_wall={wall_fps:.1f} fps_loop={avg_fps:.1f} "
                    f"tracks={len(fusion.tracks)} interval={args.detect_interval}"
                )
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    return 0


def configure_opencv(args: argparse.Namespace) -> None:
    cv2.setUseOptimized(True)
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)


def load_calibration(path: Optional[str]) -> Optional[CameraCalibration]:
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"Calibration file does not exist: {path}")
    return CameraCalibration.from_npz(path)


def full_frame_roi(frame) -> BBox:
    height, width = frame.shape[:2]
    return clip_bbox((0, 0, width, height), width, height)


if __name__ == "__main__":
    raise SystemExit(main())
