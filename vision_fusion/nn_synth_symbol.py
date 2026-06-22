"""符号分类器训练数据:渲染符号 marker,切 4 格自动打标签。

与 nn_synth_tri_digit 同结构,差异:
  - marker 用 symbol_marker.render_marker_symbol(底边黑条定向 + 几何符号);
  - 切格用 slice_cells(mask_tri=False, mask_bottom=True);
  - scene 采集真实样本:overlay 小角度 → warp → orient_by_edge 旋正 → 切格。
输出 datasets/symbol_crops/<char>/m*.png,char ∈ {0-9, X}(逻辑标签)。

Usage:
    python -m vision_fusion.nn_synth_symbol --id-max 999 --aug 12 --scene 18 --output datasets/symbol_crops
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .symbol_marker import render_marker_symbol, marker_chars
from .digit_detect_tri import slice_cells, CHARS, _ensure_gray, orient_by_edge
from .digit_detect import warp_square
from .nn_augment import random_degrade
from .nn_synth_ui import overlay_marker, load_settings, CANVAS_W, CANVAS_H

SETTINGS = "symbol_marker_settings.json"


def _synth_params() -> dict:
    s = load_settings()
    return dict(brightness=s.get("brightness", 100.0), contrast=s.get("contrast", 0.5),
                blur_sigma=s.get("blur_sigma", 2.5), scatter_sigma=s.get("scatter_sigma", 5.0),
                opacity=s.get("opacity", 0.8), levels_black=s.get("levels_black", 0.0),
                levels_white=s.get("levels_white", 255.0), levels_gamma=s.get("levels_gamma", 1.0))


def _render_params() -> dict:
    """符号 marker 渲染参数;有 symbol_marker_settings.json 就用,否则用默认。"""
    p = Path(SETTINGS)
    if not p.exists():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    keys = ("pixels", "border_ratio", "bottom_extra_ratio", "col_gap_ratio",
            "row_gap_ratio", "sym_size_ratio", "stroke_ratio")
    return {k: s[k] for k in keys if k in s}


def _slice(sq):
    return slice_cells(sq, mask_tri=False, mask_bottom=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize symbol-cell crops.")
    ap.add_argument("--output", default="datasets/symbol_crops")
    ap.add_argument("--id-min", type=int, default=0)
    ap.add_argument("--id-max", type=int, default=999)
    ap.add_argument("--aug", type=int, default=12, help="每格退化增广张数(干净渲染派生)。")
    ap.add_argument("--scene", type=int, default=18, help="每 marker 采集真实样本(overlay→warp→定向→切格)。")
    ap.add_argument("--archive", default=None, help="存几张切格校验图的目录。")
    args = ap.parse_args()

    rparams = _render_params()
    synth = _synth_params()
    out = Path(args.output)
    for ch in CHARS:
        (out / ch).mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    counts = {ch: 0 for ch in CHARS}
    ad = Path(args.archive) if args.archive else None
    if ad:
        ad.mkdir(parents=True, exist_ok=True)
    archived = 0

    for mid in range(args.id_min, args.id_max + 1):
        chars = marker_chars(mid)
        marker = render_marker_symbol(mid, **rparams)
        for ch, cell in zip(chars, _slice(marker)):
            cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), cell)
            counts[ch] += 1
            for _ in range(args.aug):
                deg = random_degrade(cell, rng)
                cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), deg)
                counts[ch] += 1
        # 采集真实样本:小角度 overlay → warp → 边定向旋正 → 切格(mask 底条)
        for _ in range(args.scene):
            base = rng.randint(95, 178)
            scene = np.full((CANVAS_H, CANVAS_W), base, np.uint8)
            scene = np.clip(scene.astype(np.float32) +
                            np.random.normal(0, rng.uniform(4, 16), scene.shape), 0, 255).astype(np.uint8)
            ang = rng.uniform(-8, 8)
            sc = rng.uniform(0.22, 0.55)
            scene, corners, _ = overlay_marker(scene, marker, CANVAS_W // 2, CANVAS_H // 2,
                                               ang, sc, **synth)
            sq = _ensure_gray(warp_square(scene, corners, size=200))
            if sq.size == 0:
                continue
            sq = orient_by_edge(sq)        # 底条旋到底部
            for ch, cell in zip(chars, _slice(sq)):
                cv2.imwrite(str(out / ch / f"m{mid:03d}_s{counts[ch]:05d}.png"), cell)
                counts[ch] += 1
        if ad and archived < 6:
            vis = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(ad / f"sym_{mid:03d}_{chars}.png"), vis)
            archived += 1

    print("per-class counts:", counts)
    print(f"output: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
