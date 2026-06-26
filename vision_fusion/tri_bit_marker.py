"""Tri-Bit Marker 生成器:4×4 格 marker,左上 2×2 大三角定向 + 12 格数据(无校验)。

规格:
  - 外框:四周等宽黑框
  - 内部 4×4 = 16 格
  - 左上 2×2 (格 0,1,4,5): 一个大三角(直角右下,斜边左上→右下) = 定向
  - 其余 12 格(2,3, 6,7, 8,9,10,11, 12,13,14,15): 数据位
  - id 容量: 2^12 = 4096
  - 无校验:任何 bit 错 = 静默错读
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np


N_DATA = 12                     # 12 个数据格
MAX_ID = (1 << N_DATA) - 1     # 4095

# 数据格在 4×4 网格中的索引(行扫描,跳过左上 2x2 = 索引 0,1,4,5)
DATA_CELL_IDX = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# 默认布局参数
TRI_BIT_DEFAULTS = dict(
    border_ratio=0.08,
    gap_ratio=0.0,
    fill_ratio=1.0,
)


def id_to_bits(marker_id: int) -> list[int]:
    """id (0..4095) -> 12 bit 列表(MSB first),1=黑 0=白。"""
    if not 0 <= marker_id <= MAX_ID:
        raise ValueError(f"id 超范围 0..{MAX_ID}")
    return [(marker_id >> (N_DATA - 1 - i)) & 1 for i in range(N_DATA)]


def bits_to_id(bits: list[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def encode_id(marker_id: int) -> list[int]:
    """兼容旧接口:无校验直接返回 12 bit。"""
    return id_to_bits(marker_id)


def decode_bits(bits: list[int]) -> int:
    """12 bit -> id。无校验,直接读。"""
    if len(bits) != N_DATA:
        return -1
    return bits_to_id(bits)


def render_tri_bit_marker(marker_id: int, *, pixels: int = 600, **overrides) -> np.ndarray:
    """渲染 tri-bit marker (灰度,白底255墨0)。"""
    cfg = {**TRI_BIT_DEFAULTS, **overrides}
    p = pixels
    B = max(2, int(p * cfg["border_ratio"]))
    gap = cfg["gap_ratio"]
    fill = cfg["fill_ratio"]
    img = np.full((p, p), 0, np.uint8)
    inner = p - 2 * B
    cv2.rectangle(img, (B, B), (p - 1 - B, p - 1 - B), 255, -1)

    cell = inner / 4.0
    g = int(cell * gap / 2)
    bits12 = id_to_bits(marker_id)

    # 左上 2x2 大三角(覆盖整个 2x2 区域的右下半)
    tri_x0 = int(B + 0 * cell) + g; tri_y0 = int(B + 0 * cell) + g
    tri_x1 = int(B + 2 * cell) - g; tri_y1 = int(B + 2 * cell) - g
    tw, th = tri_x1 - tri_x0, tri_y1 - tri_y0
    px = int(tw * (1 - fill) / 2); py = int(th * (1 - fill) / 2)
    fx0, fy0 = tri_x0 + px, tri_y0 + py
    fx1, fy1 = tri_x1 - px, tri_y1 - py
    # 直角在右下:点 = 右上,右下,左下
    tri_pts = np.array([[fx1, fy0], [fx1, fy1], [fx0, fy1]], np.int32)
    cv2.fillPoly(img, [tri_pts], 0)

    # 12 个数据格
    for di, idx in enumerate(DATA_CELL_IDX):
        row, col = idx // 4, idx % 4
        cx0 = int(B + col * cell) + g
        cy0 = int(B + row * cell) + g
        cx1 = int(B + (col + 1) * cell) - g
        cy1 = int(B + (row + 1) * cell) - g
        cw = cx1 - cx0; ch = cy1 - cy0
        pad_x = int(cw * (1 - fill) / 2)
        pad_y = int(ch * (1 - fill) / 2)
        fx0d = cx0 + pad_x; fy0d = cy0 + pad_y
        fx1d = cx1 - pad_x; fy1d = cy1 - pad_y
        if bits12[di]:
            cv2.rectangle(img, (fx0d, fy0d), (fx1d, fy1d), 0, -1)
    return img


def _ids_from_args(args):
    if args.range:
        return list(range(args.range[0], args.range[1] + 1))
    return list(args.id or [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Tri-Bit marker generator (4x4, 2x2 big triangle + 12-bit data).")
    ap.add_argument("--id", type=int, nargs="*", help=f"单个或多个 id (0..{MAX_ID})")
    ap.add_argument("--range", type=int, nargs=2, metavar=("LO", "HI"), help="闭区间批量")
    ap.add_argument("--output", default="tri_bit_markers", help="输出目录")
    ap.add_argument("--pixels", type=int, default=600)
    args = ap.parse_args()
    ids = _ids_from_args(args)
    if not ids:
        print("需要 --id 或 --range")
        return 1
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        cv2.imwrite(str(out / f"tribit_{mid:05d}.png"),
                    render_tri_bit_marker(mid, pixels=args.pixels))
    print(f"wrote {len(ids)} tri-bit markers -> {out.resolve()}  (容量 2^12={MAX_ID+1}, 无校验)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
