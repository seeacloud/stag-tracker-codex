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

# 默认布局(归一化)。正方形被等宽黑线分成 4 等大白格。线宽 = 外框 = 内部分隔(同步)。
# 定向:内部十字"缺口"——竖分隔只画上半、横分隔只画右半,交汇成 ┘ 拐角(旋转后唯一)。
SYM_DEFAULTS = dict(
    line_ratio=0.08,            # 黑线宽(外框=分隔,同步)
    padding_ratio=0.04,         # 裁切框相对白格内壁的等距内缩
    sym_fill=0.72,              # 符号占裁切框比例(<1 留白,不超格)
    stroke_ratio=0.18,
    round_cap=False,
)


def _white_cells(line_ratio):
    """4 个等大白格框 [(x0,y0,x1,y1)*4] [TL,TR,BL,BR]。布局 line|cell|line|cell|line。"""
    L = line_ratio
    cell = (1 - 3 * L) / 2.0
    seg = [(L, L + cell), (2 * L + cell, 2 * L + 2 * cell)]
    return [(seg[c][0], seg[r][0], seg[c][1], seg[r][1]) for r in range(2) for c in range(2)]


def cell_boxes(line_ratio=SYM_DEFAULTS["line_ratio"],
               padding_ratio=SYM_DEFAULTS["padding_ratio"], **_) -> list[tuple]:
    """裁切框 = 白格等距内缩 padding。唯一真相源,render/slice/定向共用。[TL,TR,BL,BR]。"""
    return [(x0 + padding_ratio, y0 + padding_ratio, x1 - padding_ratio, y1 - padding_ratio)
            for x0, y0, x1, y1 in _white_cells(line_ratio)]


def marker_chars(marker_id: int) -> str:
    return f"{marker_id:03d}" + checksum_char(marker_id)


def render_marker_symbol(marker_id: int, *, pixels: int = 300, **overrides) -> np.ndarray:
    """符号 marker(灰度,白底255墨0,正方形)。全白底→画黑外框+L形缺口分隔→放符号。
    定向缺口:竖分隔只上半(中心→上)、横分隔只右半(中心→右)→ ┘ 拐角。"""
    cfg = {**SYM_DEFAULTS, **overrides}
    p = pixels
    L = max(2, int(p * cfg["line_ratio"]))
    cells = _white_cells(cfg["line_ratio"])
    img = np.full((p, p), 0, np.uint8)                             # 全黑
    # 整个内区填白(留外框 L),其余三格连通;只左上格被封闭。
    cv2.rectangle(img, (L, L), (p - 1 - L, p - 1 - L), 255, -1)
    m = p // 2
    # 只画"上竖 + 左横"两段黑线 → 封闭左上格,其余连通(定向)。
    cv2.rectangle(img, (m - L // 2, L), (m + L // 2, m + L // 2), 0, -1)   # 上竖(下探到中心交汇)
    cv2.rectangle(img, (L, m - L // 2), (m + L // 2, m + L // 2), 0, -1)   # 左横(右探到中心交汇)
    chars = marker_chars(marker_id)
    for ch, (nx0, ny0, nx1, ny1) in zip(chars, cells):            # 符号放白格正中(上下左右居中)
        cx = (nx0 + nx1) / 2 * p; cy = (ny0 + ny1) / 2 * p
        box = (nx1 - nx0) * p - 2 * cfg["padding_ratio"] * p       # 裁切框边长
        side = max(12, int(box * cfg["sym_fill"]))
        sym = draw_symbol(ch, size=side, stroke=max(2, int(cfg["stroke_ratio"] * side)),
                          round_cap=cfg["round_cap"])
        gx0 = int(cx - side / 2); gy0 = int(cy - side / 2)
        roi = img[gy0:gy0 + side, gx0:gx0 + side]
        if roi.shape == sym.shape:
            img[gy0:gy0 + side, gx0:gx0 + side] = np.minimum(roi, sym)
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
