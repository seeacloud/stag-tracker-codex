import cv2
import numpy as np
import pytest

from vision_fusion.digit_marker_tri import generate_marker_tri
from vision_fusion.digit_detect_tri import find_triangle_corner, orient_by_triangle

# cv2 旋转 → 三角从 TL 移到的角(0=TL 1=TR 2=BR 3=BL)
CASES = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 1),
    (cv2.ROTATE_180, 2),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 3),
]


def _degrade(img, sigma=3.0, lo=90, hi=170):
    """模拟实拍：高斯模糊 + 压低对比到窄灰阶。"""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return (lo + (blur.astype(np.float32) / 255.0) * (hi - lo)).astype(np.uint8)


def test_find_triangle_corner_all_rotations():
    gen = generate_marker_tri(83, pixels=200)
    for rot, expected in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        corner, conf = find_triangle_corner(img)
        assert corner == expected
        assert conf > 0


def test_orient_by_triangle_returns_to_top_left():
    gen = generate_marker_tri(83, pixels=200)
    for rot, _ in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        oriented, ok = orient_by_triangle(img)
        assert ok
        assert find_triangle_corner(oriented)[0] == 0


def test_find_triangle_corner_survives_blur():
    gen = generate_marker_tri(83, pixels=200)
    for rot, expected in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        assert find_triangle_corner(_degrade(img))[0] == expected
