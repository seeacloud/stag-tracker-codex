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
from .digit_detect_tri import TRI_COL_GAP, TRI_ROW_GAP
from .symbol_set import draw_symbol

# 2×2 符号区垂直中心(归一化)。<0.5 表示上移,给底部黑条让位。
# render 与识别(slice_cells)共用此值,保证切格几何一致。
SYM_CENTER_Y = 0.44


def marker_chars(marker_id: int) -> str:
    return f"{marker_id:03d}" + checksum_char(marker_id)


def render_marker_symbol(marker_id: int, *, pixels: int = 300, border_ratio: float = 0.06,
                         bottom_extra_ratio: float = 0.06, col_gap_ratio: float = TRI_COL_GAP,
                         row_gap_ratio: float = TRI_ROW_GAP, sym_size_ratio: float = 0.30,
                         stroke_ratio: float = 0.16, round_cap: bool = True) -> np.ndarray:
    """渲染符号 marker(灰度,背景 255 墨 0)。正方形画布。

    底条在正方形内底部(厚 = border + bottom_extra);2×2 符号区**整体上移**避让底条,
    垂直中心在 SYM_CENTER_Y(<0.5),与 slice_cells 的切格中心一致。横向居中 0.5。
    """
    p = pixels
    be = int(p * bottom_extra_ratio)
    img = np.full((p, p), 255, np.uint8)                            # 正方形画布
    b = int(p * border_ratio)
    cv2.rectangle(img, (0, 0), (p - 1, p - 1), 0, -1)               # 全黑
    cv2.rectangle(img, (b, b), (p - 1 - b, p - 1 - b), 255, -1)     # 挖白 → 黑边框
    cv2.rectangle(img, (b, p - 1 - b - be), (p - 1 - b, p - 1 - b), 0, -1)  # 底条(区内底部)

    cx = (p - 1) / 2.0                                              # 横向居中
    cy = SYM_CENTER_Y * (p - 1)                                    # 纵向上移(避让底条)
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
