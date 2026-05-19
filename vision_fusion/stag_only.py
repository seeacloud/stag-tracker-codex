from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional, TextIO

import cv2
import numpy as np

from .fusion import FusionTracker
from .models import BBox, StagCandidate, StagObservation, Track, clip_bbox
from .optical_flow import OpticalFlowTracker
from .preprocess import EnhanceConfig
from .screen_mapper import ScreenMapper, draw_screen_observations, draw_screen_tracks
from .stag_detector import CameraCalibration, StagDetector
from .tuio_sender import TuioSender, tracks_to_tuio_objects
from .visualization import draw_candidates, draw_observations, draw_tracks
from .camera_picker import probe_cameras, print_cameras, pick_camera_gui, camera_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect STag markers only.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--stag-library", type=int, default=17, help="STag HD library number.")
    parser.add_argument("--roi-padding", type=int, default=12, help="Pixels added around the detection ROI.")
    parser.add_argument("--marker-size", type=float, default=None, help="Marker side length in meters.")
    parser.add_argument("--calibration", default=None, help="NPZ with camera_matrix and dist_coeffs.")
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip preview/output horizontally without mirroring STag detection.",
    )
    parser.add_argument("--screen-map", default=None, help="NPZ from vision_fusion.calibrate_screen.")
    parser.add_argument("--screen-output", default=None, help="Optional warped screen output video path.")
    parser.add_argument("--no-screen-view", action="store_true", help="Do not show the warped screen window.")
    parser.add_argument("--no-screen-roi", action="store_true", help="Do not restrict full searches to the screen map.")
    parser.add_argument("--tuio", action="store_true", help="Send TUIO 1.1 /tuio/2Dobj messages over UDP.")
    parser.add_argument("--tuio-host", default="127.0.0.1", help="TUIO UDP target host.")
    parser.add_argument("--tuio-port", type=int, default=3333, help="TUIO UDP target port.")
    parser.add_argument("--tuio-source", default="vision_fusion", help="TUIO source name.")
    parser.add_argument("--no-memory", action="store_true", help="Disable optical-flow memory tracking.")
    parser.add_argument("--detect-interval", type=int, default=1, help="Run STag detection every N frames.")
    parser.add_argument("--reacquire-interval", type=int, default=30, help="Run a full-frame search every N frames.")
    parser.add_argument("--fallback-full-interval", type=int, default=5, help="Add a full-screen/screen-map ROI every N frames while tracking.")
    parser.add_argument("--search-padding", type=int, default=80, help="Pixels around remembered tracks for STag reacquisition.")
    parser.add_argument("--max-missed", type=int, default=60, help="Drop a track after this many optical-flow misses.")
    parser.add_argument("--visual-hold", type=int, default=6, help="Keep marker visually recognized for this many missed detection frames.")
    parser.add_argument("--flow-points", type=int, default=120, help="Maximum optical-flow points per marker.")
    parser.add_argument("--flow-min-points", type=int, default=6, help="Minimum valid optical-flow points to trust tracking.")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="Display smoothing weight. Lower is steadier; 1 disables smoothing.")
    parser.add_argument("--smooth-deadband", type=float, default=1.5, help="Ignore display movement smaller than this many pixels.")
    parser.add_argument("--smooth-snap", type=float, default=70.0, help="Snap display to raw track after a large movement.")
    parser.add_argument("--smooth-step", type=float, default=0.05, help="Keyboard step for live smoothing changes.")
    parser.add_argument("--show", action="store_true", help="Show annotated frames.")
    parser.add_argument("--output", default=None, help="Optional annotated output video path.")
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
    parser.add_argument("--log-every", type=int, default=120, help="Print average FPS every N frames. 0 disables logs.")
    parser.add_argument("--log-jsonl", default=None, help="Append per-frame JSON lines with detection state to this file.")
    parser.add_argument("--list-cameras", action="store_true", help="List detected cameras and exit.")
    parser.add_argument("--pick-camera", action="store_true", help="Show a camera picker window before starting.")
    parser.add_argument("--enhance-clahe", action="store_true", help="Apply CLAHE before STag detection (helps low-light/low-contrast).")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit. Higher boosts contrast more.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size (NxN).")
    parser.add_argument(
        "--enhance-sharpen",
        action="store_true",
        help="Apply unsharp-mask sharpening before STag (helps motion/focus blur).",
    )
    parser.add_argument("--sharpen-amount", type=float, default=1.0, help="Unsharp-mask strength (0=off, 1.0 default, 2.0 strong).")
    parser.add_argument("--sharpen-radius", type=float, default=1.2, help="Unsharp-mask Gaussian sigma (pixels). Larger smooths a bigger neighbourhood.")
    parser.add_argument("--sharpen-threshold", type=int, default=0, help="Skip pixels whose |orig-blur| <= threshold (suppresses noise amplification).")
    parser.add_argument(
        "--camera-exposure",
        type=float,
        default=None,
        help="DSHOW exposure value, log2 seconds (e.g. -7 for ~1/128s). Lower = shorter exposure = less motion blur.",
    )
    parser.add_argument(
        "--scales",
        default="1.0",
        help="Comma-separated detection scales, e.g. '0.75,1.0,1.5'. Multi-scale helps small/large markers.",
    )
    parser.add_argument(
        "--roi-min-short-side",
        type=int,
        default=0,
        help="If a detection ROI's short side is below this many pixels, super-resolve it before STag (0 disables).",
    )
    parser.add_argument(
        "--show-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw blue boxes around STag-localized but undecoded quads.",
    )
    return parser.parse_args()


def parse_scales(spec: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise SystemExit(f"--scales contains invalid number: {token!r}") from exc
        if value <= 0:
            raise SystemExit(f"--scales must be positive, got {value}")
        values.append(value)
    if not values:
        return (1.0,)
    return tuple(values)


def main() -> int:
    args = parse_args()

    if args.list_cameras or args.pick_camera:
        backend = camera_backend(args.camera_backend if args.camera_backend != "any" else "dshow")
        cameras = probe_cameras(backend=backend)
        if args.list_cameras:
            print_cameras(cameras)
            if not args.pick_camera:
                return 0
        if args.pick_camera:
            if not cameras:
                raise SystemExit("No cameras detected.")
            picked = pick_camera_gui(cameras)
            if picked is None:
                raise SystemExit("Camera selection cancelled.")
            args.source = str(picked)
            print(f"Selected camera index {picked}")

    capture = open_capture(args)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Video source opened but did not return a frame.")

    height, width = frame.shape[:2]
    writer = build_writer(args.output, capture, width, height) if args.output else None
    screen_mapper = ScreenMapper.load(args.screen_map) if args.screen_map else None
    screen_writer = (
        build_fixed_writer(args.screen_output, capture, screen_mapper.output_size)
        if args.screen_output and screen_mapper is not None
        else None
    )
    tuio_sender = (
        TuioSender(args.tuio_host, args.tuio_port, args.tuio_source)
        if args.tuio
        else None
    )
    calibration = load_calibration(args.calibration)
    detector = StagDetector(
        library_hd=args.stag_library,
        marker_size=args.marker_size,
        calibration=calibration,
        roi_padding=args.roi_padding,
        enhance=EnhanceConfig(
            clahe=args.enhance_clahe,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            sharpen=args.enhance_sharpen,
            sharpen_amount=args.sharpen_amount,
            sharpen_radius=args.sharpen_radius,
            sharpen_threshold=args.sharpen_threshold,
        ),
        scales=parse_scales(args.scales),
        roi_min_short_side=args.roi_min_short_side,
    )
    flow = OpticalFlowTracker(
        max_corners=args.flow_points,
        min_points=args.flow_min_points,
        use_affine=True,
    )
    fusion = FusionTracker(
        flow=flow,
        max_missed=args.max_missed,
        smooth_alpha=args.smooth_alpha,
        smooth_deadband=args.smooth_deadband,
        smooth_snap=args.smooth_snap,
    )

    frame_index = 0
    start_time = time.perf_counter()
    last_time = start_time
    fps_window: deque[float] = deque(maxlen=120)
    previous_gray = None
    jsonl_file: Optional[TextIO] = None
    if args.log_jsonl:
        Path(args.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(args.log_jsonl, "w", encoding="utf-8", buffering=1)

    try:
        while ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            predicted = False
            updated = False
            if not args.no_memory and previous_gray is not None and fusion.tracks:
                fusion.predict(previous_gray, gray)
                predicted = True

            observations = []
            should_detect = frame_index % max(1, args.detect_interval) == 0
            if should_detect:
                rois = detection_rois(frame, fusion.tracks, args, frame_index, screen_mapper)
                observations = detector.detect(frame, rois=rois)
                if not args.no_memory:
                    fusion.update(gray, [], observations)
                    updated = True

            if not args.no_memory and predicted and not updated:
                fusion.record_predicted_history()

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_window.append(fps)

            display_frame = maybe_mirror(frame, args.mirror)
            display_tracks = (
                mirror_tracks_for_display(fusion.tracks, frame.shape[1])
                if args.mirror
                else fusion.tracks
            )
            display_observations = (
                mirror_observations_for_display(observations, frame.shape[1])
                if args.mirror
                else observations
            )
            display_candidates = (
                mirror_candidates_for_display(detector.last_candidates, frame.shape[1])
                if args.mirror
                else detector.last_candidates
            )

            annotated = display_frame.copy()
            if screen_mapper is not None:
                screen_mapper.draw_source_outline(annotated)
            if args.show_candidates:
                draw_candidates(annotated, display_candidates)
            if args.no_memory:
                draw_observations(annotated, display_observations)
                active_count = len(display_observations)
            else:
                draw_tracks(annotated, display_tracks, visual_hold=args.visual_hold)
                active_count = len(display_tracks)
            draw_status(
                annotated,
                frame_index,
                fps,
                active_count,
                not args.no_memory,
                fusion.smooth_alpha,
                fusion.smooth_deadband,
            )

            if writer is not None:
                writer.write(annotated)

            screen_view = None
            if screen_mapper is not None:
                screen_input = display_frame if args.mirror else frame
                screen_view = screen_mapper.warp(screen_input)
                if args.no_memory:
                    draw_screen_observations(screen_view, display_observations, screen_mapper)
                else:
                    draw_screen_tracks(
                        screen_view,
                        display_tracks,
                        screen_mapper,
                        visual_hold=args.visual_hold,
                    )
                draw_screen_status(screen_view, fps, active_count)
                if screen_writer is not None:
                    screen_writer.write(screen_view)

            if tuio_sender is not None:
                tuio_sender.send(
                    tracks_to_tuio_objects(
                        fusion.tracks,
                        frame.shape,
                        screen_mapper,
                    )
                )

            if args.show:
                cv2.imshow("STag only", annotated)
                if screen_view is not None and not args.no_screen_view:
                    cv2.imshow("Screen view", screen_view)
                key = cv2.waitKeyEx(1)
                if handle_key(key, fusion, args.smooth_step):
                    break

            ok, frame = capture.read()
            previous_gray = gray
            frame_index += 1
            if args.log_every and frame_index % args.log_every == 0 and fps_window:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                wall_fps = frame_index / elapsed
                loop_fps = sum(fps_window) / len(fps_window)
                print(
                    f"frame={frame_index} fps_wall={wall_fps:.1f} "
                    f"fps_loop={loop_fps:.1f} active={active_count} "
                    f"observed={len(observations)}"
                )
            if jsonl_file is not None:
                record = {
                    "frame": frame_index,
                    "fps": round(fps, 2),
                    "observed_ids": [int(o.marker_id) for o in observations],
                    "candidates": [
                        [int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])]
                        for c in detector.last_candidates
                    ],
                    "tracks": [
                        {
                            "id": int(t.marker_id),
                            "bbox": [int(v) for v in t.bbox],
                            "missed": int(t.missed),
                            "stag_missed": int(getattr(t, "stag_missed", 0)),
                            "recognized": bool(
                                getattr(t, "stag_missed", 0) <= args.visual_hold
                            ),
                        }
                        for t in fusion.tracks
                    ],
                }
                jsonl_file.write(json.dumps(record) + "\n")
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if screen_writer is not None:
            screen_writer.release()
        if tuio_sender is not None:
            tuio_sender.send([])
            tuio_sender.close()
        if jsonl_file is not None:
            jsonl_file.close()
        if args.show:
            cv2.destroyAllWindows()

    return 0


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source.isdigit():
        capture = cv2.VideoCapture(int(args.source), camera_backend(args.camera_backend))
        configure_camera(capture, args)
        return capture
    return cv2.VideoCapture(args.source)


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
    if args.camera_exposure is not None:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        capture.set(cv2.CAP_PROP_EXPOSURE, args.camera_exposure)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def build_writer(path: str, capture: cv2.VideoCapture, width: int, height: int) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def build_fixed_writer(
    path: str,
    capture: cv2.VideoCapture,
    output_size: tuple[int, int],
) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)


def load_calibration(path: Optional[str]) -> Optional[CameraCalibration]:
    if not path:
        return None
    if not Path(path).exists():
        raise SystemExit(f"Calibration file does not exist: {path}")
    return CameraCalibration.from_npz(path)


def full_frame_roi(frame) -> BBox:
    height, width = frame.shape[:2]
    return clip_bbox((0, 0, width, height), width, height)


def maybe_mirror(frame, enabled: bool):
    if not enabled:
        return frame
    return cv2.flip(frame, 1)


def mirror_tracks_for_display(tracks: list[Track], width: int) -> list[Track]:
    return [mirror_track_for_display(track, width) for track in tracks]


def mirror_track_for_display(track: Track, width: int) -> Track:
    return replace(
        track,
        bbox=mirror_bbox(track.bbox, width),
        corners=mirror_points(track.corners, width) if track.corners is not None else None,
        display_bbox=(
            mirror_float_bbox(track.display_bbox, width)
            if track.display_bbox is not None
            else None
        ),
        display_corners=(
            mirror_points(track.display_corners, width)
            if track.display_corners is not None
            else None
        ),
        history=[(width - 1 - x, y) for x, y in track.history],
    )


def mirror_observations_for_display(
    observations: list[StagObservation],
    width: int,
) -> list[StagObservation]:
    return [
        StagObservation(
            marker_id=observation.marker_id,
            corners=mirror_points(observation.corners, width),
            bbox=mirror_bbox(observation.bbox, width),
            pose=observation.pose,
        )
        for observation in observations
    ]


def mirror_candidates_for_display(
    candidates: list[StagCandidate],
    width: int,
) -> list[StagCandidate]:
    return [
        StagCandidate(
            corners=mirror_points(candidate.corners, width),
            bbox=mirror_bbox(candidate.bbox, width),
        )
        for candidate in candidates
    ]


def mirror_bbox(bbox: BBox, width: int) -> BBox:
    x, y, w, h = bbox
    return width - x - w, y, w, h


def mirror_float_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(width - x - w), float(y), float(w), float(h)


def mirror_points(points: np.ndarray, width: int) -> np.ndarray:
    mirrored = np.asarray(points, dtype=np.float32).copy()
    mirrored[..., 0] = width - 1 - mirrored[..., 0]
    return mirrored


def detection_rois(
    frame,
    tracks,
    args: argparse.Namespace,
    frame_index: int,
    screen_mapper: Optional[ScreenMapper],
) -> list[BBox]:
    if (
        args.no_memory
        or not tracks
        or (
            args.reacquire_interval
            and args.reacquire_interval > 0
            and frame_index % args.reacquire_interval == 0
        )
    ):
        if screen_mapper is not None and not args.no_screen_roi:
            roi = screen_mapper.source_bbox(frame.shape, padding=args.search_padding)
            if args.mirror:
                roi = mirror_bbox(roi, frame.shape[1])
            return [roi]
        return [full_frame_roi(frame)]

    height, width = frame.shape[:2]
    rois = [
        clip_bbox(track.bbox, width, height, padding=args.search_padding)
        for track in tracks
    ]
    if args.fallback_full_interval and frame_index % args.fallback_full_interval == 0:
        if screen_mapper is not None and not args.no_screen_roi:
            roi = screen_mapper.source_bbox(frame.shape, padding=args.search_padding)
            if args.mirror:
                roi = mirror_bbox(roi, frame.shape[1])
            rois.append(roi)
        else:
            rois.append(full_frame_roi(frame))
    return rois


def draw_status(
    frame,
    frame_index: int,
    fps: float,
    active_count: int,
    memory_enabled: bool,
    smooth_alpha: float,
    smooth_deadband: float,
) -> None:
    mode = "STag memory" if memory_enabled else "STag only"
    label = "tracks" if memory_enabled else "markers"
    text = (
        f"{mode} | frame {frame_index} | fps {fps:.1f} | {label} {active_count} "
        f"| alpha {smooth_alpha:.2f} dead {smooth_deadband:.1f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (8, 8), (640, 34), (40, 40, 40), -1)
    cv2.putText(frame, text, (14, 27), font, 0.5, (245, 245, 245), 1, cv2.LINE_AA)


def draw_screen_status(frame, fps: float, active_count: int) -> None:
    text = f"Screen view | fps {fps:.1f} | tracks {active_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = min(frame.shape[1] - 1, 360)
    cv2.rectangle(frame, (8, 8), (width, 34), (40, 40, 40), -1)
    cv2.putText(frame, text, (14, 27), font, 0.5, (245, 245, 245), 1, cv2.LINE_AA)


def handle_key(key: int, fusion: FusionTracker, step: float) -> bool:
    if key < 0:
        return False
    if key in (27, ord("q"), ord("Q")):
        return True

    # Arrow constants from cv2.waitKeyEx on Windows, with small-code fallbacks.
    up_keys = {2490368, 82, ord("w"), ord("W")}
    down_keys = {2621440, 84, ord("s"), ord("S")}
    left_keys = {2424832, 81, ord("a"), ord("A")}
    right_keys = {2555904, 83, ord("d"), ord("D")}

    if key in up_keys:
        fusion.smooth_alpha = clamp(fusion.smooth_alpha - step, 0.05, 1.0)
        print(f"smooth alpha={fusion.smooth_alpha:.2f} deadband={fusion.smooth_deadband:.1f}")
    elif key in down_keys:
        fusion.smooth_alpha = clamp(fusion.smooth_alpha + step, 0.05, 1.0)
        print(f"smooth alpha={fusion.smooth_alpha:.2f} deadband={fusion.smooth_deadband:.1f}")
    elif key in left_keys:
        fusion.smooth_deadband = clamp(fusion.smooth_deadband - 0.5, 0.0, 10.0)
        print(f"smooth alpha={fusion.smooth_alpha:.2f} deadband={fusion.smooth_deadband:.1f}")
    elif key in right_keys:
        fusion.smooth_deadband = clamp(fusion.smooth_deadband + 0.5, 0.0, 10.0)
        print(f"smooth alpha={fusion.smooth_alpha:.2f} deadband={fusion.smooth_deadband:.1f}")
    return False


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    raise SystemExit(main())
