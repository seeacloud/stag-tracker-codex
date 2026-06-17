"""数字 Marker 解码器（digit_marker_tri 配套）：YOLO-OBB 定位 + 内角暗度三角定向
+ RapidOCR 读 4 位 + 加权 mod11+X 校验。

Usage:
    python -m vision_fusion.digit_detect_tri --source 0
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .digit_marker_tri import checksum_char

# 旋转映射：把三角所在角旋到左上。0=TL 1=TR 2=BR 3=BL。
_ROT_TO_TL = {
    0: None,
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}


def find_triangle_corner(square: np.ndarray, b_ratio: float = 0.07,
                         k_ratio: float = 0.12) -> tuple[int, float]:
    """4 个内角缝隙各取贴角小块，积分暗度最大者=黑三角所在角。

    返回 (corner, conf)。conf=(最暗-次暗)/(最暗) 作相对 margin。
    积分暗度天然低通，模糊不改变哪个角更黑（形状糊掉也无妨）。
    """
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
    s = gray.shape[0]
    b = int(s * b_ratio)
    k = max(4, int(s * k_ratio))
    patches = [
        gray[b:b + k, b:b + k],                  # TL
        gray[b:b + k, s - b - k:s - b],          # TR
        gray[s - b - k:s - b, s - b - k:s - b],  # BR
        gray[s - b - k:s - b, b:b + k],          # BL
    ]
    dark = [float((255.0 - p.astype(np.float32)).mean()) for p in patches]
    order = sorted(range(4), key=lambda i: dark[i], reverse=True)
    top, second = dark[order[0]], dark[order[1]]
    conf = (top - second) / (top + 1e-6)
    return order[0], conf


def orient_by_triangle(square: np.ndarray,
                       min_conf: float = 0.15) -> tuple[np.ndarray, bool]:
    """把黑三角旋到左上。margin 不足 → 返回原图 + False（定向存疑）。"""
    corner, conf = find_triangle_corner(square)
    if conf < min_conf:
        return square, False
    rot = _ROT_TO_TL[corner]
    return (square if rot is None else cv2.rotate(square, rot)), True
