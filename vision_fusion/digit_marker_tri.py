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
