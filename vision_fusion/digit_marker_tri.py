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
    draw.polygon([(0, 0), (cut, 0), (0, cut)], fill=255)      # 左上切角

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

