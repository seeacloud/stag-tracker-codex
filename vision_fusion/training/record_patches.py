"""Record warped quad patches from live video for CNN training.

Usage:
    python -m vision_fusion.training.record_patches --output training_data/ [tracker args...]

Runs the normal detection pipeline and saves:
- Clear patches (STag decoded): {output}/{marker_id}/{frame}_{conf:.2f}.png
- Blurry patches (tracked but undecoded): {output}/{marker_id}_blurry/{frame}.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CANONICAL_SIZE = 128
WARP_PADDING = 24
TOTAL_SIZE = CANONICAL_SIZE + 2 * WARP_PADDING

DST_PTS = np.array([
    [WARP_PADDING, WARP_PADDING],
    [WARP_PADDING + CANONICAL_SIZE, WARP_PADDING],
    [WARP_PADDING + CANONICAL_SIZE, WARP_PADDING + CANONICAL_SIZE],
    [WARP_PADDING, WARP_PADDING + CANONICAL_SIZE],
], dtype=np.float32)


def warp_quad_to_patch(frame: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    src_pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if src_pts.shape != (4, 2):
        return None
    M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
    warped = cv2.warpPerspective(frame, M, (TOTAL_SIZE, TOTAL_SIZE))
    if warped.ndim == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped


def save_patch(output_dir: Path, marker_id: int, frame_idx: int,
               patch: np.ndarray, blurry: bool = False, confidence: float = 1.0) -> None:
    if blurry:
        subdir = output_dir / f"{marker_id}_blurry"
    else:
        subdir = output_dir / str(marker_id)
    subdir.mkdir(parents=True, exist_ok=True)
    if blurry:
        filename = f"{frame_idx:06d}.png"
    else:
        filename = f"{frame_idx:06d}_{confidence:.2f}.png"
    cv2.imwrite(str(subdir / filename), patch)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record training patches from live video.")
    parser.add_argument("--output", required=True, help="Output directory for patches.")
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--stag-library", type=int, default=17)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=60)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-backend", choices=("any", "dshow", "msmf"), default="msmf")
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=unlimited).")
    parser.add_argument("--save-every", type=int, default=3, help="Save a patch every N frames (avoid near-duplicates).")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import stag
    from ..preprocess import apply_gamma
    from ..camera_picker import camera_backend

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

    # Track which marker IDs are at which locations (for labeling blurry candidates)
    track_positions: dict[int, tuple[np.ndarray, int]] = {}  # marker_id -> (last_corners, last_frame)

    frame_idx = 0
    saved_count = 0
    print(f"Recording patches to {output_dir}/ — press 'q' to stop")
    print(f"Saving every {args.save_every} frames to avoid duplicates")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = apply_gamma(gray, args.gamma)

            try:
                corners, ids, rejected = stag.detectMarkers(enhanced, args.stag_library)
            except TypeError:
                corners, ids, rejected = stag.detectMarkers(enhanced, libraryHD=args.stag_library)

            # Save clear detections
            if ids is not None and len(ids) > 0 and frame_idx % args.save_every == 0:
                for i, marker_id_arr in enumerate(ids):
                    marker_id = int(marker_id_arr[0]) if hasattr(marker_id_arr, '__len__') else int(marker_id_arr)
                    c = corners[i].reshape(4, 2)
                    patch = warp_quad_to_patch(gray, c)
                    if patch is not None:
                        save_patch(output_dir, marker_id, frame_idx, patch, blurry=False, confidence=1.0)
                        saved_count += 1
                    track_positions[marker_id] = (c, frame_idx)

            # Save blurry candidates (label from nearest tracked marker)
            if rejected is not None and len(rejected) > 0 and frame_idx % args.save_every == 0:
                for quad in rejected:
                    quad_pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
                    quad_center = quad_pts.mean(axis=0)
                    best_id = None
                    best_dist = 150.0  # max pixel distance to associate
                    for mid, (last_c, last_f) in track_positions.items():
                        if frame_idx - last_f > 90:  # stale track
                            continue
                        track_center = last_c.mean(axis=0)
                        dist = float(np.linalg.norm(quad_center - track_center))
                        if dist < best_dist:
                            best_dist = dist
                            best_id = mid
                    if best_id is not None:
                        patch = warp_quad_to_patch(gray, quad_pts)
                        if patch is not None:
                            save_patch(output_dir, best_id, frame_idx, patch, blurry=True)
                            saved_count += 1

            # Display
            display = frame.copy()
            if ids is not None:
                stag.drawDetectedMarkers(display, corners, ids)
            info = f"frame={frame_idx} saved={saved_count} detected={len(ids) if ids is not None else 0}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Record Training Patches", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone. Saved {saved_count} patches across {frame_idx} frames.")
    print(f"Output: {output_dir}/")
    # Print summary
    dirs = sorted(output_dir.iterdir()) if output_dir.exists() else []
    for d in dirs:
        if d.is_dir():
            count = len(list(d.glob("*.png")))
            print(f"  {d.name}: {count} patches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
