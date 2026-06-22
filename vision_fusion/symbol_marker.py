"""符号 marker 渲染:外框 + 底边加宽黑条(定向) + 2×2 符号格。

布局复刻 digit_marker_tri.generate_marker_tri,但:
  - 定向标记从"左上内角黑三角"改为"底边加宽黑条"(信号更强、抗模糊、不占内部);
  - 格内从数字字形改为 symbol_set.draw_symbol 的几何符号。
逻辑标签仍是 0-9X,复用 checksum_char。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .digit_marker_tri import checksum_char
from .symbol_set import draw_symbol


def marker_chars(marker_id: int) -> str:
    return f"{marker_id:03d}" + checksum_char(marker_id)


def render_marker_symbol(marker_id: int, *, pixels: int = 300, border_ratio: float = 0.06,
                         bottom_extra_ratio: float = 0.06, pad_ratio: float = 0.08,
                         col_gap_ratio: float = 0.38, row_gap_ratio: float = 0.42,
                         sym_size_ratio: float = 0.30, stroke_ratio: float = 0.16,
                         round_cap: bool = True) -> np.ndarray:
    """渲染符号 marker(灰度,背景 255 墨 0)。底边为加宽黑条作定向。"""
    p = pixels
    img = np.full((p, p), 255, np.uint8)
    b = int(p * border_ratio)
    cv2.rectangle(img, (0, 0), (p - 1, p - 1), 0, -1)               # 全黑
    cv2.rectangle(img, (b, b), (p - 1 - b, p - 1 - b), 255, -1)     # 挖白 → 黑边框
    be = int(p * bottom_extra_ratio)                               # 底边加宽黑条(定向)
    cv2.rectangle(img, (b, p - 1 - b - be), (p - 1 - b, p - 1 - b), 0, -1)

    pad = int(p * pad_ratio)
    lo, hi = b + pad, p - 1 - b - be - pad
    cx = cy = (lo + hi) / 2.0
    col = p * col_gap_ratio
    row = p * row_gap_ratio
    centers = [(cx - col / 2, cy - row / 2), (cx + col / 2, cy - row / 2),
               (cx - col / 2, cy + row / 2), (cx + col / 2, cy + row / 2)]

    chars = marker_chars(marker_id)
    cell = max(16, int(p * sym_size_ratio))
    stroke = max(2, int(stroke_ratio * cell))
    for ch, (gx, gy) in zip(chars, centers):
        sym = draw_symbol(ch, size=cell, stroke=stroke, round_cap=round_cap)
        x0, y0 = int(gx - cell / 2), int(gy - cell / 2)
        roi = img[y0:y0 + cell, x0:x0 + cell]
        if roi.shape == sym.shape:
            img[y0:y0 + cell, x0:x0 + cell] = np.minimum(roi, sym)  # 墨色(取暗)叠加
    return img


def _ids_from_args(args) -> list[int]:
    if args.range:
        return list(range(args.range[0], args.range[1] + 1))
    return list(args.id or [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Symbol marker generator.")
    ap.add_argument("--id", type=int, nargs="*", help="单个或多个 ID")
    ap.add_argument("--range", type=int, nargs=2, metavar=("LO", "HI"), help="闭区间批量")
    ap.add_argument("--output", default="symbol_markers", help="输出目录")
    ap.add_argument("--pixels", type=int, default=600)
    args = ap.parse_args()
    ids = _ids_from_args(args)
    if not ids:
        print("需要 --id 或 --range")
        return 1
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        g = render_marker_symbol(mid, pixels=args.pixels)
        cv2.imwrite(str(out / f"symbol_{mid:03d}_{checksum_char(mid)}.png"), g)
    print(f"wrote {len(ids)} symbol markers → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
