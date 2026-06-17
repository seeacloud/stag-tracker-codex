"""数字 Marker 生成器：左上切角定向 + 2×2 数字 + 加权 mod 11(X) 校验。

设计见 docs/superpowers/specs/2026-06-17-digit-marker-tri-generator-design.md
不复用旧的 digit_marker.checksum(求和 mod 10)——本版用加权 mod 11 抓换位错。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "C:/Windows/Fonts/consolab.ttf"


def checksum_char(marker_id: int) -> str:
    """加权 mod 11 校验位；结果 10 按 ISBN-10 风格返回 'X'。"""
    d = f"{marker_id:03d}"
    c = (1 * int(d[0]) + 2 * int(d[1]) + 3 * int(d[2])) % 11
    return "X" if c == 10 else str(c)


def generate_marker_tri(
    marker_id: int,
    *,
    pixels: int = 600,
    border_ratio: float = 0.07,
    chamfer_ratio: float = 0.18,
    pad_ratio: float = 0.06,
    col_gap_ratio: float = 0.34,
    row_gap_ratio: float = 0.34,
    font_path: str = DEFAULT_FONT,
    font_size_ratio: float = 0.28,
    stroke_ratio: float = 0.0,
) -> np.ndarray:
    """切角边框 + 2×2 数字 marker（灰度 ndarray，背景 255 墨色 0）。"""
    p = pixels
    img = Image.new("L", (p, p), 255)
    draw = ImageDraw.Draw(img)

    b = int(p * border_ratio)
    draw.rectangle([0, 0, p - 1, p - 1], fill=0)              # 全黑
    draw.rectangle([b, b, p - 1 - b, p - 1 - b], fill=255)    # 挖白内部 → 黑边框
    cut = int(p * chamfer_ratio)
    # 左上内角黑三角（定向标记）：内框左上角填黑，使白窗口左上成 45° 黑切角。
    draw.polygon([(b, b), (b + cut, b), (b, b + cut)], fill=0)

    pad = int(p * pad_ratio)
    lo, hi = b + pad, p - 1 - b - pad
    cx = cy = (lo + hi) / 2.0
    col = p * col_gap_ratio
    row = p * row_gap_ratio
    centers = [
        (cx - col / 2, cy - row / 2),  # TL d1
        (cx + col / 2, cy - row / 2),  # TR d2
        (cx - col / 2, cy + row / 2),  # BL d3
        (cx + col / 2, cy + row / 2),  # BR check
    ]

    text = f"{marker_id:03d}{checksum_char(marker_id)}"
    size = max(8, int(p * font_size_ratio))
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()
    stroke_w = max(0, round(stroke_ratio * size))

    for glyph, (gx, gy) in zip(text, centers):
        bb = font.getbbox(glyph, stroke_width=stroke_w)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((gx - tw / 2 - bb[0], gy - th / 2 - bb[1]), glyph,
                  fill=0, font=font, stroke_width=stroke_w, stroke_fill=0)

    return np.array(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chamfer-oriented 2x2 digit markers (weighted mod-11 checksum).")
    parser.add_argument("--id", type=int, default=None, help="Single marker ID (0-999).")
    parser.add_argument("--range", nargs=2, type=int, default=None,
                        metavar=("START", "END"), help="Inclusive ID range to batch-export.")
    parser.add_argument("--output", type=str, default="digit_markers_tri",
                        help="Output directory (created if missing).")
    parser.add_argument("--pixels", type=int, default=600)
    parser.add_argument("--border-ratio", type=float, default=0.07)
    parser.add_argument("--chamfer-ratio", type=float, default=0.18)
    parser.add_argument("--pad-ratio", type=float, default=0.06)
    parser.add_argument("--col-gap-ratio", type=float, default=0.34)
    parser.add_argument("--row-gap-ratio", type=float, default=0.34)
    parser.add_argument("--font-path", type=str, default=DEFAULT_FONT)
    parser.add_argument("--font-size-ratio", type=float, default=0.28)
    parser.add_argument("--stroke-ratio", type=float, default=0.0)
    return parser.parse_args()


def _ids_from_args(args: argparse.Namespace) -> list[int]:
    if args.id is not None:
        return [args.id]
    if args.range is not None:
        return list(range(args.range[0], args.range[1] + 1))
    return list(range(0, 100))


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ids = _ids_from_args(args)
    for mid in ids:
        arr = generate_marker_tri(
            mid, pixels=args.pixels, border_ratio=args.border_ratio,
            chamfer_ratio=args.chamfer_ratio, pad_ratio=args.pad_ratio,
            col_gap_ratio=args.col_gap_ratio, row_gap_ratio=args.row_gap_ratio,
            font_path=args.font_path, font_size_ratio=args.font_size_ratio,
            stroke_ratio=args.stroke_ratio,
        )
        Image.fromarray(arr).save(str(out / f"digit_{mid:03d}_{checksum_char(mid)}.png"))
    print(f"Generated {len(ids)} markers in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


