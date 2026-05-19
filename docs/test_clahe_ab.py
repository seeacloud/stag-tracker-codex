"""Single-frame A/B test: raw vs CLAHE for STag detection.

Captures one frame from the picked camera, runs STag on:
- raw image
- CLAHE (BGR via LAB)
- CLAHE on grayscale only

Saves all three variants + result PNGs for visual inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from vision_fusion.camera_picker import camera_backend, default_camera_index, probe_cameras
from vision_fusion.preprocess import apply_clahe
from vision_fusion.stag_detector import StagDetector


def detect(detector: StagDetector, image: np.ndarray) -> tuple[list, np.ndarray]:
    obs = detector.detect(image)
    annotated = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for o in obs:
        pts = o.corners.astype(np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        cv2.putText(annotated, f"id={o.marker_id}", tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return obs, annotated


def main() -> int:
    out_dir = Path("docs/test-screenshots/clahe_ab")
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = probe_cameras(backend=camera_backend("dshow"))
    if not cams:
        print("No camera detected.")
        return 1
    idx = default_camera_index(cams)
    print(f"Using camera idx {idx}: {next(c.name for c in cams if c.index == idx)}")

    cap = cv2.VideoCapture(idx, camera_backend("dshow"))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # warm up + grab one stable frame
    for _ in range(15):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Failed to capture frame.")
        return 1

    cv2.imwrite(str(out_dir / "00_raw.png"), frame)

    detector = StagDetector(library_hd=17)

    raw_obs, raw_ann = detect(detector, frame)
    cv2.imwrite(str(out_dir / "01_raw_result.png"), raw_ann)

    clahe_bgr = apply_clahe(frame, clip=2.0, grid=8)
    cv2.imwrite(str(out_dir / "02_clahe_bgr.png"), clahe_bgr)
    cb_obs, cb_ann = detect(detector, clahe_bgr)
    cv2.imwrite(str(out_dir / "03_clahe_bgr_result.png"), cb_ann)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    clahe_gray_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_dir / "04_clahe_gray.png"), clahe_gray_bgr)
    cg_obs, cg_ann = detect(detector, clahe_gray_bgr)
    cv2.imwrite(str(out_dir / "05_clahe_gray_result.png"), cg_ann)

    print(f"raw           : {len(raw_obs)} markers  ids={[o.marker_id for o in raw_obs]}")
    print(f"clahe-bgr/lab : {len(cb_obs)} markers  ids={[o.marker_id for o in cb_obs]}")
    print(f"clahe-gray    : {len(cg_obs)} markers  ids={[o.marker_id for o in cg_obs]}")
    print(f"\nArtifacts in: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
