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

from .digit_marker_tri import generate_marker_tri, checksum_char
from .digit_detect_tri import slice_cells, cell_centers, CHARS, CELL_HALF
from .nn_augment import random_degrade

SETTINGS = "digit_marker_tri_settings.json"


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
    ap.add_argument("--aug", type=int, default=20, help="每格退化增广张数。")
    ap.add_argument("--archive", default=None, help="存几张切格校验图的目录。")
    args = ap.parse_args()

    params = _render_params()
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
