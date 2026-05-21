from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional, TextIO

import cv2
import numpy as np

from .fusion import FusionTracker
from .kalman_predictor import KalmanPredictor
from .candidate_recovery import CandidateRecovery
from .models import BBox, StagCandidate, StagObservation, Track, clip_bbox
from .optical_flow import OpticalFlowTracker
from .preprocess import EnhanceConfig
from .async_detector import AsyncDetector
from .process_camera import ProcessCamera
from .screen_mapper import ScreenMapper, draw_screen_observations, draw_screen_tracks
from .stag_detector import CameraCalibration, PassConfig, StagDetector
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
    parser.add_argument(
        "--predictor",
        choices=("kalman", "flow"),
        default="kalman",
        help="Prediction method between detections. 'kalman' is lightweight (<1ms), 'flow' uses optical flow (~40ms). Default kalman.",
    )
    parser.add_argument("--detect-interval", type=int, default=1, help="Submit detection every N frames. Default 1 = every frame (async pipeline handles throughput).")
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
    parser.add_argument("--raw-output", default=None, help="Optional unannotated original-frame video (use for clean A/B replay).")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 runs until the source ends.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera capture width (default 1280 = 720p).")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera capture height (default 720 = 720p).")
    parser.add_argument("--camera-fps", type=float, default=60, help="Requested camera capture FPS. Default 60.")
    parser.add_argument("--camera-fourcc", default="MJPG", help="Requested camera FOURCC. Default MJPG enables 720p@30 on most USB webcams.")
    parser.add_argument(
        "--camera-backend",
        choices=("any", "dshow", "msmf"),
        default="msmf",
        help="OpenCV camera backend on Windows. Default 'msmf' achieves 720p@60fps on this project's USB cam.",
    )
    parser.add_argument("--log-every", type=int, default=120, help="Print average FPS every N frames. 0 disables logs.")
    parser.add_argument("--log-jsonl", default=None, help="Append per-frame JSON lines with detection state to this file.")
    parser.add_argument("--list-cameras", action="store_true", help="List detected cameras and exit.")
    parser.add_argument("--pick-camera", action="store_true", help="Show a camera picker window before starting.")
    parser.add_argument(
        "--enhance-clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE before STag detection (helps low-light/low-contrast). Default on so the 1-pass fallback is also a good config.",
    )
    parser.add_argument("--clahe-clip", type=float, default=3.5, help="CLAHE clip limit. Higher boosts contrast more. Default 3.5 is the verified best-1-pass value.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size (NxN).")
    parser.add_argument(
        "--enhance-sharpen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply unsharp-mask sharpening before STag. Default off (net negative on 1-pass). Multi-pass uses the per-pass :on/:off field instead.",
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
        "--gamma",
        type=float,
        default=0.6,
        help="Gamma correction applied before detection. <1 brightens (compensates dark frames). Default 0.6 is empirically optimal for MSMF 720p.",
    )
    parser.add_argument(
        "--scales",
        default="0.75,1.0,1.5",
        help="Comma-separated detection scales for the 1-pass fallback path, e.g. '0.75,1.0,1.5'. Default is the verified best-1-pass triplet. Ignored when --detect-passes is non-empty.",
    )
    parser.add_argument(
        "--roi-min-short-side",
        type=int,
        default=140,
        help="If a detection ROI's short side is below this many pixels, super-resolve it before STag (0 disables). Default 140 is the verified best-1-pass value. Ignored when --detect-passes is non-empty.",
    )
    parser.add_argument(
        "--show-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw blue boxes around STag-localized but undecoded quads.",
    )
    parser.add_argument(
        "--detect-passes",
        default="3.5:0.75,1.0,1.5:140:off;4.5:0.6,1.0,2.0:100:on;4.0:0.3,0.5,1.0:80:5.0,4.0;3.5:0.25,0.4,0.7:60:2.0,2.0:4",
        help=(
            "Multi-pass detection spec, semicolon-separated. Each pass is "
            "'clahe_clip:scales:roi_min_short_side[:sharpen[:deconv]]'. "
            "Default is a 4-pass config: baseline + aggressive + deblur + deconv. "
            "Pass 3 uses strong sharpening (5.0, radius 4.0) for moderate blur. "
            "Pass 4 uses Wiener deconvolution (disc radius 4) for severe defocus. "
            "Use 'off' for clahe_clip to disable CLAHE in that pass. "
            "Sharpen field: 'off', 'on', a number, or 'amount,radius'. "
            "Deconv field: integer disc radius for Wiener deconvolution (0=off). "
            "Pass empty string ('') to fall back to the legacy single-pass path."
        ),
    )
    parser.add_argument(
        "--pass-workers",
        type=int,
        default=12,
        help="Process pool size for parallel detection. Default 12 matches 4-pass x 3-scale.",
    )
    parser.add_argument(
        "--expected-ids",
        default="",
        help=(
            "Comma-separated marker ids that should be in the scene, e.g. '0,1,2,10,150'. "
            "When set, multi-pass detection skips later (slower) passes whenever the "
            "first pass already finds all of them. Recovers FPS in static scenes "
            "without sacrificing recognition rate."
        ),
    )
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Path to trained CNN classifier model (.pt) for blurry marker recovery. Enables GPU-accelerated marker ID prediction on rejected candidates.",
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for CNN classifier predictions. Default 0.7.",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to trained YOLO marker locator model (.pt). Finds markers that STag's quad detection misses.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.4,
        help="YOLO detection confidence threshold. Default 0.4.",
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


def parse_expected_ids(spec: str) -> Optional[set[int]]:
    """Parse '0,1,2,10,150' -> {0,1,2,10,150}. Empty -> None (adaptive disabled)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError as exc:
            raise SystemExit(f"--expected-ids contains non-integer: {token!r}") from exc
    return out or None


def parse_passes(spec: str, base_enhance: EnhanceConfig) -> list[PassConfig]:
    """Parse a multi-pass spec like '3.5:0.75,1.0,1.5:140;4.5:1.0,2.0:100'.

    Each pass: 'clahe_clip:scales:roi_min_short_side[:sharpen[:deconv]]'.
    Use 'off' for clahe_clip to disable CLAHE in that pass.
    Optional 4th field (sharpen):
      - omitted        → inherit base_enhance.sharpen (back-compat)
      - 'off'          → sharpen disabled for this pass
      - 'on'           → sharpen enabled, amount inherited from --sharpen-amount
      - <float>        → sharpen enabled with that amount
      - 'amount,radius' → sharpen with specific amount and radius
    Optional 5th field (deconv):
      - omitted or '0' → no deconvolution
      - <int>          → Wiener deconvolution with disc PSF of that radius

    Returns [] if spec is empty (caller falls back to legacy single-pass path).
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    passes: list[PassConfig] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) not in (3, 4, 5):
            raise SystemExit(
                f"--detect-passes pass {chunk!r} must have 3-5 fields "
                f"'clahe_clip:scales:roi_min_short_side[:sharpen[:deconv]]'"
            )
        clip_token = parts[0].strip()
        scales_token = parts[1].strip()
        min_token = parts[2].strip()
        sharpen_token = parts[3].strip() if len(parts) >= 4 else None
        deconv_token = parts[4].strip() if len(parts) >= 5 else None

        if clip_token.lower() == "off":
            clahe_on = False
            clahe_clip = base_enhance.clahe_clip
        else:
            try:
                clahe_clip = float(clip_token)
            except ValueError as exc:
                raise SystemExit(
                    f"--detect-passes clahe_clip must be a number or 'off', got {clip_token!r}"
                ) from exc
            clahe_on = True
        try:
            roi_min = int(min_token)
        except ValueError as exc:
            raise SystemExit(
                f"--detect-passes roi_min_short_side must be int, got {min_token!r}"
            ) from exc

        sharpen_on = base_enhance.sharpen
        sharpen_amount = base_enhance.sharpen_amount
        sharpen_radius = base_enhance.sharpen_radius
        if sharpen_token is not None:
            t = sharpen_token.lower()
            if t == "off":
                sharpen_on = False
            elif t == "on":
                sharpen_on = True
            elif "," in sharpen_token:
                parts_s = sharpen_token.split(",")
                try:
                    sharpen_amount = float(parts_s[0])
                    sharpen_radius = float(parts_s[1])
                except (ValueError, IndexError) as exc:
                    raise SystemExit(
                        f"--detect-passes sharpen must be 'on'/'off', a number, or 'amount,radius', got {sharpen_token!r}"
                    ) from exc
                sharpen_on = True
            else:
                try:
                    sharpen_amount = float(sharpen_token)
                except ValueError as exc:
                    raise SystemExit(
                        f"--detect-passes sharpen must be 'on'/'off' or a number, got {sharpen_token!r}"
                    ) from exc
                sharpen_on = True

        deconv_radius = 0
        if deconv_token is not None and deconv_token != "0":
            try:
                deconv_radius = int(deconv_token)
            except ValueError as exc:
                raise SystemExit(
                    f"--detect-passes deconv must be an integer radius, got {deconv_token!r}"
                ) from exc

        passes.append(
            PassConfig(
                enhance=replace(
                    base_enhance,
                    clahe=clahe_on,
                    clahe_clip=clahe_clip,
                    sharpen=sharpen_on,
                    sharpen_amount=sharpen_amount,
                    sharpen_radius=sharpen_radius,
                    deconv_radius=deconv_radius,
                ),
                scales=parse_scales(scales_token),
                roi_min_short_side=max(0, roi_min),
            )
        )
    if not passes:
        raise SystemExit("--detect-passes parsed to zero passes")
    return passes


class ThreadedCamera:
    """Non-blocking camera reader. Grabs frames in a background thread."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._cap = capture
        self._frame: Optional[np.ndarray] = None
        self._new_frame = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._running = False
                return
            with self._lock:
                self._frame = frame
                self._new_frame = True

    def read(self) -> tuple[bool, Optional[np.ndarray], bool]:
        """Returns (ok, frame, is_new). frame is always the latest; is_new indicates fresh."""
        with self._lock:
            if not self._running and self._frame is None:
                return False, None, False
            is_new = self._new_frame
            self._new_frame = False
            return True, self._frame, is_new

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


def _stabilize_recovered_corners(
    marker_id: int, new_corners: np.ndarray, tracks: list, alpha: float = 0.3
) -> np.ndarray:
    """Blend CNN-recovered corners with existing track corners to reduce jitter."""
    new_pts = np.asarray(new_corners, dtype=np.float32).reshape(4, 2)
    for track in tracks:
        if track.marker_id == marker_id and track.corners is not None:
            old_pts = np.asarray(track.corners, dtype=np.float32).reshape(4, 2)
            return old_pts + alpha * (new_pts - old_pts)
    return new_pts


def _overlaps_existing(
    bbox: BBox, observations: list, tracks: list, iou_threshold: float = 0.05
) -> bool:
    """Reject if bbox overlaps or is contained within any existing detection/track."""
    from .models import bbox_iou
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    for obs in observations:
        if bbox_iou(bbox, obs.bbox) > iou_threshold:
            return True
        # Center-point containment check
        ox, oy, ow, oh = obs.bbox
        if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
            return True
    for track in tracks:
        if bbox_iou(bbox, track.bbox) > iou_threshold:
            return True
        tx, ty, tw, th = track.bbox
        if tx <= cx <= tx + tw and ty <= cy <= ty + th:
            return True
    return False


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
    raw_writer = build_writer(args.raw_output, capture, width, height) if args.raw_output else None
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
    base_enhance = EnhanceConfig(
        gamma=args.gamma,
        clahe=args.enhance_clahe,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        sharpen=args.enhance_sharpen,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        sharpen_threshold=args.sharpen_threshold,
    )
    detect_passes = parse_passes(args.detect_passes, base_enhance)
    expected_ids = parse_expected_ids(args.expected_ids)
    detector = StagDetector(
        library_hd=args.stag_library,
        marker_size=args.marker_size,
        calibration=calibration,
        roi_padding=args.roi_padding,
        enhance=base_enhance,
        scales=parse_scales(args.scales),
        roi_min_short_side=args.roi_min_short_side,
        passes=detect_passes or None,
        pass_workers=args.pass_workers,
        expected_ids=expected_ids,
    )
    async_det = AsyncDetector(
        library_hd=args.stag_library,
        roi_padding=args.roi_padding,
        passes=detect_passes or detector.passes,
        workers=args.pass_workers,
    )
    if args.predictor == "kalman":
        kalman = KalmanPredictor()
        flow = None
    else:
        kalman = None
        flow = OpticalFlowTracker(
            max_corners=args.flow_points,
            min_points=args.flow_min_points,
            use_affine=True,
        )
    fusion = FusionTracker(
        flow=flow,
        kalman=kalman,
        predictor=args.predictor,
        max_missed=args.max_missed,
        smooth_alpha=args.smooth_alpha,
        smooth_deadband=args.smooth_deadband,
        smooth_snap=args.smooth_snap,
    )
    classifier = None
    if args.classifier_model:
        from .cnn_classifier import MarkerClassifier
        classifier = MarkerClassifier(
            model_path=args.classifier_model,
            confidence_threshold=args.classifier_threshold,
        )
        print(f"CNN classifier loaded: {args.classifier_model}")
    recovery = CandidateRecovery(library_hd=args.stag_library, classifier=classifier)
    yolo_locator = None
    if args.yolo_model:
        from .yolo_locator import YoloLocator
        yolo_locator = YoloLocator(
            model_path=args.yolo_model,
            confidence=args.yolo_confidence,
        )
        print(f"YOLO locator loaded: {args.yolo_model}")

    frame_index = 0
    start_time = time.perf_counter()
    last_time = start_time
    fps_window: deque[float] = deque(maxlen=120)
    previous_gray = None
    jsonl_file: Optional[TextIO] = None
    if args.log_jsonl:
        Path(args.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(args.log_jsonl, "w", encoding="utf-8", buffering=1)

    profile_accum: dict[str, float] = {
        "gray": 0.0, "flow": 0.0, "detect": 0.0,
        "fusion": 0.0, "viz": 0.0, "display": 0.0,
    }
    profile_count = 0
    profile_interval = args.log_every or 120

    use_threaded = False
    proc_cam: Optional[ProcessCamera] = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated = frame.copy()
    screen_view = None
    active_count = 0
    cam_frame_count = 0
    cam_fps_start = time.perf_counter()
    cam_fps: float = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            got_new_frame = True

            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t1 = time.perf_counter()

            if got_new_frame:
                cam_frame_count += 1
            cam_elapsed = t1 - cam_fps_start
            if cam_elapsed >= 0.5:
                cam_fps = cam_frame_count / cam_elapsed
                cam_frame_count = 0
                cam_fps_start = t1

            predicted = False
            updated = False
            if not args.no_memory and fusion.tracks:
                if args.predictor == "kalman":
                    fusion.predict()
                    predicted = True
                elif previous_gray is not None:
                    fusion.predict(previous_gray, gray)
                    predicted = True
            t2 = time.perf_counter()

            observations = []
            # Poll for completed detection results FIRST (non-blocking)
            async_result = async_det.try_get_results()
            if async_result is not None:
                observations, new_candidates = async_result
                # Learn templates from successful detections
                for obs in observations:
                    recovery.learn_template(obs.marker_id, frame, obs.corners)

                # Filter out candidates inside already-confirmed markers
                if new_candidates:
                    from .models import bbox_from_points as _bfp2
                    new_candidates = [
                        c for c in new_candidates
                        if not _overlaps_existing(_bfp2(c.corners), observations, fusion.tracks)
                    ]
                detector.last_candidates = new_candidates

            # YOLO locator: find ALL markers (including blurry ones STag missed)
            # Then try STag decode on each region; if fail → CNN classify
            if yolo_locator is not None and async_result is not None:
                existing_bboxes = [obs.bbox for obs in observations]
                existing_bboxes += [t.bbox for t in fusion.tracks]
                yolo_regions = yolo_locator.locate(frame, existing_bboxes)

                for corners, bbox in yolo_regions:
                    # Step 1: try STag decode on this region
                    import stag as _stag
                    x, y, w, h = bbox
                    pad = 10
                    rx1 = max(0, x - pad)
                    ry1 = max(0, y - pad)
                    rx2 = min(gray.shape[1], x + w + pad)
                    ry2 = min(gray.shape[0], y + h + pad)
                    roi_gray = gray[ry1:ry2, rx1:rx2]

                    stag_id = None
                    if roi_gray.size > 0:
                        try:
                            s_corners, s_ids, _ = _stag.detectMarkers(roi_gray, args.stag_library)
                        except TypeError:
                            s_corners, s_ids, _ = _stag.detectMarkers(roi_gray, libraryHD=args.stag_library)
                        if s_ids is not None and len(s_ids) > 0:
                            stag_id = int(s_ids[0][0]) if hasattr(s_ids[0], '__len__') else int(s_ids[0])
                            # Offset corners back to full frame coords
                            det_corners = s_corners[0].reshape(4, 2) + np.array([rx1, ry1], dtype=np.float32)
                            corners = det_corners

                    if stag_id is not None:
                        from .models import bbox_from_points
                        obs = StagObservation(
                            marker_id=stag_id,
                            corners=corners,
                            bbox=bbox_from_points(corners),
                            pose=None,
                        )
                        observations.append(obs)
                    elif classifier is not None:
                        # Step 2: CNN classify
                        src_pts = corners.reshape(4, 2).astype(np.float32)
                        dst_pts = np.array([
                            [0, 0], [128, 0], [128, 128], [0, 128]
                        ], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        patch = cv2.warpPerspective(gray, M, (128, 128))
                        predictions = classifier.classify([patch])
                        if predictions and predictions[0][0] is not None:
                            marker_id, conf = predictions[0]
                            obs = StagObservation(
                                marker_id=marker_id,
                                corners=corners,
                                bbox=bbox,
                                pose=None,
                            )
                            observations.append(obs)

            if async_result is not None:
                if not args.no_memory:
                    fusion.update(gray, [], observations)
                    updated = True

            # Then submit next frame for detection (non-blocking)
            should_detect = got_new_frame and frame_index % max(1, args.detect_interval) == 0
            if should_detect:
                rois = detection_rois(frame, fusion.tracks, args, frame_index, screen_mapper)
                async_det.submit(frame, rois)
            t3 = time.perf_counter()

            if not args.no_memory and predicted and not updated:
                fusion.record_predicted_history()
            t4 = time.perf_counter()

            now = t4
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps_window.append(fps)

            need_redraw = True
            if need_redraw:
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
                    cam_fps=cam_fps,
                )

            if need_redraw:
                if writer is not None:
                    writer.write(annotated)
                if raw_writer is not None:
                    raw_writer.write(frame)

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
            t5 = time.perf_counter()

            if need_redraw and tuio_sender is not None:
                tuio_sender.send(
                    tracks_to_tuio_objects(
                        fusion.tracks,
                        frame.shape,
                        screen_mapper,
                    )
                )

            if args.show and need_redraw:
                cv2.imshow("STag only", annotated)
                if screen_view is not None and not args.no_screen_view:
                    cv2.imshow("Screen view", screen_view)
            if args.show:
                key = cv2.waitKeyEx(1)
                if handle_key(key, fusion, args.smooth_step):
                    break
            t6 = time.perf_counter()

            profile_accum["gray"] += t1 - t0
            profile_accum["flow"] += t2 - t1
            profile_accum["detect"] += t3 - t2
            profile_accum["fusion"] += t4 - t3
            profile_accum["viz"] += t5 - t4
            profile_accum["display"] += t6 - t5
            profile_count += 1

            previous_gray = gray
            frame_index += 1
            if profile_interval and frame_index % profile_interval == 0 and fps_window:
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                wall_fps = frame_index / elapsed
                loop_fps = sum(fps_window) / len(fps_window)
                n = max(profile_count, 1)
                avg = {k: v / n * 1000 for k, v in profile_accum.items()}
                total_ms = sum(avg.values())
                print(
                    f"frame={frame_index} fps_wall={wall_fps:.1f} "
                    f"fps_loop={loop_fps:.1f} cam_fps={cam_fps:.1f} active={active_count} "
                    f"observed={len(observations)}"
                )
                print(
                    f"  profile (avg ms): "
                    f"gray={avg['gray']:.1f} flow={avg['flow']:.1f} "
                    f"detect={avg['detect']:.1f} fusion={avg['fusion']:.1f} "
                    f"viz={avg['viz']:.1f} display={avg['display']:.1f} "
                    f"total={total_ms:.1f}"
                )
                profile_accum = {k: 0.0 for k in profile_accum}
                profile_count = 0
            if jsonl_file is not None:
                record = {
                    "frame": frame_index,
                    "fps": round(fps, 2),
                    "observed_ids": [int(o.marker_id) for o in observations],
                    "skipped_passes": int(getattr(detector, "last_skipped_passes", 0)),
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
        async_det.close()
        detector.close()
        capture.release()
        if writer is not None:
            writer.release()
        if raw_writer is not None:
            raw_writer.release()
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
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        capture.set(cv2.CAP_PROP_EXPOSURE, args.camera_exposure)
    elif args.camera_backend == "msmf":
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        capture.set(cv2.CAP_PROP_EXPOSURE, -4)
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
    cam_fps: float = 0.0,
) -> None:
    mode = "STag memory" if memory_enabled else "STag only"
    label = "tracks" if memory_enabled else "markers"
    cam_str = f" | cam {cam_fps:.1f}" if cam_fps > 0 else ""
    text = (
        f"{mode} | frame {frame_index} | fps {fps:.1f}{cam_str} | {label} {active_count} "
        f"| alpha {smooth_alpha:.2f} dead {smooth_deadband:.1f}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width = cv2.getTextSize(text, font, 0.5, 1)[0][0]
    cv2.rectangle(frame, (8, 8), (text_width + 22, 34), (40, 40, 40), -1)
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
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
