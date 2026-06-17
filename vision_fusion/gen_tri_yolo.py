"""tri marker（内角黑三角）YOLO-OBB 检测训练数据生成器。

基于 gen_notch_yolo：marker 来源换成 digit_marker_tri.generate_marker_tri（直接渲染，
不依赖 PNG 目录/命名），复用 nn_synth_ui.overlay_marker 的降质合成 + OBB 角点标注。
摄像头帧当背景（CLAHE+镜像，与推理一致）；无相机时用灰底+噪声兜底，不硬退出。

Usage:
    python -m vision_fusion.gen_tri_yolo --count 800 --output datasets/tri_det
    python -m vision_fusion.gen_tri_yolo --count 30 --no-camera --output datasets/tri_det_sample
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

from .nn_synth_ui import overlay_marker, load_settings, CANVAS_W, CANVAS_H


def load_markers(markers_dir: str) -> dict:
    """加载用户导出的 tri marker PNG（digit_{id:03d}_{check}.png，灰度）。

    直接用用户生成的图本身（含其 GUI 调好的字体/字号/三角等），保证合成 marker 与
    将来打印/部署的完全一致——不重渲染，避免字体路径解析/参数偏差导致字号偏小。
    """
    d = Path(markers_dir)
    markers: dict[int, np.ndarray] = {}
    for mid in range(1000):
        hits = sorted(d.glob(f"digit_{mid:03d}_*.png"))
        if hits:
            img = cv2.imread(str(hits[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                markers[mid] = img
    return markers


def _candidate_corners(cx: float, cy: float, angle_deg: float, scale: float,
                       marker_px: int) -> np.ndarray:
    """复现 overlay_marker 的四角：以 (cx,cy) 为心、边长 max(16,int(px*scale)) 的
    方块绕中心旋转 angle。画之前先拿到角点做重叠检测。"""
    s = max(16, int(marker_px * scale))
    half = s / 2.0
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    out = []
    for ox, oy in ((-half, -half), (half, -half), (half, half), (-half, half)):
        out.append([ox * cos_a - oy * sin_a + cx, ox * sin_a + oy * cos_a + cy])
    return np.array(out, dtype=np.float32)


def _inflate(corners: np.ndarray, factor: float = 1.12) -> np.ndarray:
    """绕中心放大角点，给重叠检测留安全边距（永不重叠是宪法级硬约束）。"""
    c = corners.mean(axis=0)
    return ((corners - c) * factor + c).astype(np.float32)


def _overlaps(corners: np.ndarray, placed: list[np.ndarray]) -> bool:
    """与任一已放 marker 有交叠则 True（凸四边形精确相交，带安全边距）。"""
    a = _inflate(corners)
    for q in placed:
        area, _ = cv2.intersectConvexConvex(a, _inflate(q))
        if area > 1.0:
            return True
    return False



def grab_bg(cap) -> np.ndarray:
    """相机帧（CLAHE+镜像）当背景；无相机用灰底+噪声兜底。"""
    if cap is not None and cap.isOpened():
        ok, frame = cap.read()
        if ok:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            g = cv2.flip(g, 1)
            g = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8)).apply(g)
            return cv2.resize(g, (CANVAS_W, CANVAS_H))
    base = np.full((CANVAS_H, CANVAS_W), 140, np.uint8)
    return np.clip(base.astype(np.float32) + np.random.normal(0, 12, base.shape), 0, 255).astype(np.uint8)


def yolo_obb_label(corners: np.ndarray, w: int, h: int) -> str:
    pts = np.clip(corners.reshape(-1) / np.array([w, h] * 4, np.float32), 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in pts)


def main() -> int:
    ap = argparse.ArgumentParser(description="tri marker YOLO-OBB dataset generator.")
    ap.add_argument("--count", type=int, default=800, help="场景数。")
    ap.add_argument("--output", default="datasets/tri_det")
    ap.add_argument("--markers-dir", default="digit_markers_tri",
                    help="用户导出的 tri marker PNG 目录(digit_NNN_*.png)。")
    ap.add_argument("--source", default="0")
    ap.add_argument("--id-max", type=int, default=999, help="只用 ID ≤ 此值的 marker。")
    ap.add_argument("--per-scene-min", type=int, default=6)
    ap.add_argument("--per-scene-max", type=int, default=16)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--no-camera", action="store_true", help="不开摄像头，用灰底兜底背景。")
    args = ap.parse_args()

    # 直接载入用户导出的 marker PNG（不重渲染，外观与部署一致）。
    markers = load_markers(args.markers_dir)
    available = [m for m in sorted(markers) if m <= args.id_max]
    if not available:
        print(f"ERROR: {args.markers_dir} 里没找到 digit_NNN_*.png。"
              f"先用 GUI/CLI 导出 tri marker 到该目录。")
        return 1
    print(f"loaded {len(available)} markers from {args.markers_dir}")

    s = load_settings()
    params = dict(brightness=s.get("brightness", 100.0), contrast=s.get("contrast", 0.5),
                  blur_sigma=s.get("blur_sigma", 2.5), scatter_sigma=s.get("scatter_sigma", 5.0),
                  opacity=s.get("opacity", 0.8), levels_black=s.get("levels_black", 0.0),
                  levels_white=s.get("levels_white", 255.0), levels_gamma=s.get("levels_gamma", 1.0))

    out = Path(args.output)
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    cap = None
    if not args.no_camera:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        if cap.isOpened():
            for _ in range(8):
                cap.read()
        else:
            print("WARN: camera not opened, using gray fallback background")
            cap = None

    rng = random.Random(7)
    val_every = max(1, round(1 / args.val_ratio)) if args.val_ratio > 0 else 0
    for idx in range(args.count):
        split = "val" if val_every and idx % val_every == 0 else "train"
        scene = grab_bg(cap).copy()
        labels = []
        placed: list[np.ndarray] = []          # 已放 marker 的角点，用于永不重叠检测
        target = rng.randint(args.per_scene_min, args.per_scene_max)
        for _ in range(target):
            # 拒绝采样：最多试 40 次找一个不与已放 marker 交叠的位置；找不到就放弃这一个。
            for _attempt in range(40):
                mid = rng.choice(available)
                cx = rng.randint(80, CANVAS_W - 80)
                cy = rng.randint(80, CANVAS_H - 80)
                ang = rng.uniform(-180, 180)
                sc = rng.uniform(0.18, 0.5)
                corners = _candidate_corners(cx, cy, ang, sc, markers[mid].shape[1])
                if not _overlaps(corners, placed):
                    break
            else:
                continue  # 这一个没找到空位，跳过（宁可少放也不重叠）
            scene, corners, _ = overlay_marker(scene, markers[mid], cx, cy, ang, sc, **params)
            placed.append(corners)
            labels.append(yolo_obb_label(corners, CANVAS_W, CANVAS_H))
        scene = np.clip(scene.astype(np.float32) +
                        np.random.normal(0, rng.uniform(8, 22), scene.shape), 0, 255).astype(np.uint8)
        stem = f"tri_{idx:05d}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), scene, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="ascii")
        if idx % 50 == 0:
            print(f"  {idx}/{args.count} ({split}, {len(placed)} markers)")

    if cap is not None:
        cap.release()
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['marker']\n",
        encoding="ascii")
    print(f"dataset: {out.resolve()}  scenes: {args.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
