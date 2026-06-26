"""Bit Marker 生成器:2×2 象限 bit 符号 marker。

设计(本会话验证):
  - 13 个码 = 2×2 象限"非全黑非全白"组合取前 13(码 0-12),每码 4 bit(TL,TR,BL,BR)。
  - 4 格 marker = 3 数据位 + 1 校验位,全 13 进制;外框+┘缺口定向(复用 symbol_marker 布局)。
  - 校验 c = (1·d0 + 3·d1 + 5·d2) mod 13(质数模,单数位错 100% 可检)。
  - id 容量 13³ = 2197。
识别见 bit_detect.py。之前的 symbol 系列保留不动。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, numpy as np
from .symbol_marker import _white_cells, SYM_DEFAULTS

# 13 码 bit 表(TL,TR,BL,BR):非全黑非全白的 14 组合取前 13。1=黑。
_ALL = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)
        if 0 < (a + b + c + d) < 4]
CODE_BITS = _ALL[:13]                       # 码号 0-12 -> 4 bit
N_CODES = 13
WEIGHTS = (1, 3, 5)

BIT_DEFAULTS = dict(line_ratio=SYM_DEFAULTS["line_ratio"],
                    padding_ratio=SYM_DEFAULTS["padding_ratio"],
                    sym_fill=1.0)           # 象限填满裁切框


def checksum(digits: list[int]) -> int:
    return sum(w * d for w, d in zip(WEIGHTS, digits)) % N_CODES


def id_to_codes(marker_id: int) -> list[int]:
    """id(0..2196) -> [d0,d1,d2,check] 4 个码号。"""
    if not 0 <= marker_id < N_CODES ** 3:
        raise ValueError(f"id 超范围 0..{N_CODES**3 - 1}")
    d = [(marker_id // N_CODES ** 2) % N_CODES,
         (marker_id // N_CODES) % N_CODES,
         marker_id % N_CODES]
    return d + [checksum(d)]


def codes_to_id(codes4: list[int]) -> int:
    """4 码 -> id;校验不过返回 -1。"""
    if any(not 0 <= c < N_CODES for c in codes4):
        return -1
    d, chk = codes4[:3], codes4[3]
    if checksum(d) != chk:
        return -1
    return d[0] * N_CODES ** 2 + d[1] * N_CODES + d[2]


def render_bit_marker(marker_id: int, *, pixels: int = 300, **overrides) -> np.ndarray:
    """渲染 bit marker(灰度,白底255墨0)。布局复刻 symbol_marker:黑外框+┘定向缺口+2×2格。"""
    cfg = {**BIT_DEFAULTS, **overrides}
    p = pixels
    L = max(2, int(p * cfg["line_ratio"]))
    pad = cfg["padding_ratio"]
    cells = _white_cells(cfg["line_ratio"])
    img = np.full((p, p), 0, np.uint8)
    cv2.rectangle(img, (L, L), (p - 1 - L, p - 1 - L), 255, -1)
    m = p // 2
    cv2.rectangle(img, (m - L // 2, L), (m + L // 2, m + L // 2), 0, -1)   # 上竖
    cv2.rectangle(img, (L, m - L // 2), (m + L // 2, m + L // 2), 0, -1)   # 左横
    for code, (nx0, ny0, nx1, ny1) in zip(id_to_codes(marker_id), cells):
        x0 = int(nx0 * p) + int(pad * p); y0 = int(ny0 * p) + int(pad * p)
        x1 = int(nx1 * p) - int(pad * p); y1 = int(ny1 * p) - int(pad * p)
        cw = (x1 - x0) // 2; ch = (y1 - y0) // 2
        tl, tr, bl, br = CODE_BITS[code]
        if tl: img[y0:y0 + ch, x0:x0 + cw] = 0
        if tr: img[y0:y0 + ch, x0 + cw:x1] = 0
        if bl: img[y0 + ch:y1, x0:x0 + cw] = 0
        if br: img[y0 + ch:y1, x0 + cw:x1] = 0
    return img


def _ids_from_args(args):
    if args.range:
        return list(range(args.range[0], args.range[1] + 1))
    return list(args.id or [])


def main() -> int:
    ap = argparse.ArgumentParser(description="Bit marker generator (2x2 quadrant codes).")
    ap.add_argument("--id", type=int, nargs="*", help="单个或多个 id (0..2196)")
    ap.add_argument("--range", type=int, nargs=2, metavar=("LO", "HI"), help="闭区间批量")
    ap.add_argument("--output", default="bit_markers", help="输出目录")
    ap.add_argument("--pixels", type=int, default=600)
    args = ap.parse_args()
    ids = _ids_from_args(args)
    if not ids:
        print("需要 --id 或 --range")
        return 1
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        codes = id_to_codes(mid)
        cv2.imwrite(str(out / f"bit_{mid:04d}_{'-'.join(map(str, codes))}.png"),
                    render_bit_marker(mid, pixels=args.pixels))
    print(f"wrote {len(ids)} bit markers -> {out.resolve()}  (容量 13^3={N_CODES**3})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
