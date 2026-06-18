"""tri 数字分类器训练数据：用 GUI 保存的字体/参数渲染 marker，切 4 格自动打标签。

切格复用 digit_detect_tri.slice_cells(与推理同源)。退化沿用 nn_augment.random_degrade。
输出 datasets/tri_digit_crops/<char>/m*.png，char ∈ {0-9, X}。

Usage:
    python -m vision_fusion.nn_synth_tri_digit --aug 20 --archive docs/test-screenshots/tri-slice-verify
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .digit_marker_tri import generate_marker_tri, checksum_char
from .digit_detect_tri import slice_cells, cell_centers, CHARS, CELL_HALF, _ensure_gray
from .digit_detect import warp_square
from .nn_augment import random_degrade
from .nn_synth_ui import overlay_marker, load_settings, CANVAS_W, CANVAS_H

SETTINGS = "digit_marker_tri_settings.json"


def _synth_params() -> dict:
    """nn_synth_ui 退化参数(与 gen_tri_yolo 合成场景一致)。"""
    s = load_settings()
    return dict(brightness=s.get("brightness", 100.0), contrast=s.get("contrast", 0.5),
                blur_sigma=s.get("blur_sigma", 2.5), scatter_sigma=s.get("scatter_sigma", 5.0),
                opacity=s.get("opacity", 0.8), levels_black=s.get("levels_black", 0.0),
                levels_white=s.get("levels_white", 255.0), levels_gamma=s.get("levels_gamma", 1.0))


def marker_chars(marker_id: int) -> str:
    """ID → 4 格字符串，顺序 [d1,d2,d3,check]，与 cell_centers 对应。"""
    return f"{marker_id:03d}" + checksum_char(marker_id)


def _render_params() -> dict:
    s = json.loads(Path(SETTINGS).read_text(encoding="utf-8"))
    keys = ("pixels", "font_path", "font_size_ratio", "border_ratio",
            "chamfer_ratio", "pad_ratio", "col_gap_ratio", "row_gap_ratio",
            "stroke_ratio")
    return {k: s[k] for k in keys if k in s}


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize tri digit-cell crops.")
    ap.add_argument("--output", default="datasets/tri_digit_crops")
    ap.add_argument("--id-min", type=int, default=0)
    ap.add_argument("--id-max", type=int, default=999)
    ap.add_argument("--aug", type=int, default=20, help="每格退化增广张数(干净渲染派生)。")
    ap.add_argument("--scene", type=int, default=14,
                    help="每 marker 经 overlay_marker(透视+混合+缩放)→warp→切格 的采集真实样本数。")
    ap.add_argument("--archive", default=None, help="存几张切格校验图的目录。")
    args = ap.parse_args()

    params = _render_params()
    synth = _synth_params()
    out = Path(args.output)
    for ch in CHARS:
        (out / ch).mkdir(parents=True, exist_ok=True)

    rng = random.Random(7)
    counts = {ch: 0 for ch in CHARS}
    archived = 0
    ad = Path(args.archive) if args.archive else None
    if ad:
        ad.mkdir(parents=True, exist_ok=True)

    for mid in range(args.id_min, args.id_max + 1):
        chars = marker_chars(mid)
        marker = generate_marker_tri(mid, **params)
        cells = slice_cells(marker)
        for ch, cell in zip(chars, cells):
            cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), cell)
            counts[ch] += 1
            for _ in range(args.aug):
                deg = random_degrade(cell, rng)
                cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), deg)
                counts[ch] += 1
        # 采集真实样本:小角度 overlay(透视+半透明混合+缩放上采样模糊)→ warp 拉正(仍正立)→ 切格。
        # 这补上"干净渲染+简单退化"没有的采集分布(背景透出、缩放糊、轻微透视)。
        for _ in range(args.scene):
            base = rng.randint(95, 178)
            scene = np.full((CANVAS_H, CANVAS_W), base, np.uint8)
            scene = np.clip(scene.astype(np.float32) +
                            np.random.normal(0, rng.uniform(4, 16), scene.shape), 0, 255).astype(np.uint8)
            ang = rng.uniform(-8, 8)        # 小角度→warp 后仍正立,标签可靠
            sc = rng.uniform(0.22, 0.55)    # 模拟小 marker 被上采样到 200
            scene, corners, _ = overlay_marker(scene, marker, CANVAS_W // 2, CANVAS_H // 2,
                                               ang, sc, **synth)
            sq = _ensure_gray(warp_square(scene, corners, size=200))
            if sq.size == 0:
                continue
            for ch, cell in zip(chars, slice_cells(sq)):
                cv2.imwrite(str(out / ch / f"m{mid:03d}_s{counts[ch]:05d}.png"), cell)
                counts[ch] += 1
        if ad and archived < 6:
            s = marker.shape[0]
            h = int(CELL_HALF * s)
            vis = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            for (ncx, ncy), ch in zip(cell_centers(), chars):
                cx, cy = int(ncx * s), int(ncy * s)
                cv2.rectangle(vis, (cx - h, cy - h), (cx + h, cy + h), (0, 0, 255), 2)
                cv2.putText(vis, ch, (cx - 8, cy - h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            cv2.imwrite(str(ad / f"slice_{mid:03d}_{chars}.png"), vis)
            archived += 1

    print("per-class counts:", counts)
    print(f"output: {out.resolve()}")
    if ad:
        print(f"slice-verify images: {ad.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
