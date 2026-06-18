import cv2
import numpy as np
import pytest

from vision_fusion.digit_marker_tri import generate_marker_tri
from vision_fusion.digit_detect_tri import CHARS, cell_centers, slice_cells


def test_chars_is_11_classes():
    assert CHARS == "0123456789X"


def test_cell_centers_order_and_values():
    c = cell_centers(col_gap=0.3765, row_gap=0.4176)
    assert len(c) == 4
    assert c[0] == pytest.approx((0.5 - 0.3765 / 2, 0.5 - 0.4176 / 2))  # TL d1
    assert c[1] == pytest.approx((0.5 + 0.3765 / 2, 0.5 - 0.4176 / 2))  # TR d2
    assert c[2] == pytest.approx((0.5 - 0.3765 / 2, 0.5 + 0.4176 / 2))  # BL d3
    assert c[3] == pytest.approx((0.5 + 0.3765 / 2, 0.5 + 0.4176 / 2))  # BR check


def test_slice_cells_shapes():
    g = generate_marker_tri(83, pixels=200)
    cells = slice_cells(g)
    assert len(cells) == 4
    for cell in cells:
        assert cell.shape == (64, 64)


def test_slice_cells_capture_ink():
    # 每个格子都应含黑色字形像素(白底 255, 字 0)
    g = generate_marker_tri(283, pixels=200)   # 2,8,3 + 校验, 四格都有字
    for cell in slice_cells(g):
        assert int(cell.min()) < 100
